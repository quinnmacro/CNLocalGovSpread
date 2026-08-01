"""
Wind Financial Terminal client wrapper.

Handles macOS-specific Wind Python API path detection, connection lifecycle,
EDB data fetching with retry, and graceful fallback.

Usage:
    client = WindClient()
    client.connect()
    df = client.fetch_edb(["M0017142", "M0017143"], "2018-01-01")
    client.disconnect()

Or as context manager:
    with WindClient() as client:
        df = client.fetch_edb(["M0017142"], "2018-01-01")
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging_config import get_logger

logger = get_logger(__name__)

# Known Wind Python API paths by platform
_WIND_PATHS: dict[str, list[str]] = {
    "Darwin": [
        "/Applications/Wind API.app/Contents/python",
        "/Applications/Wind Financial Terminal.app/Contents/python",
    ],
    "Windows": [
        r"C:\Wind\Wind.NET.Client\WindNET\x64",
        r"C:\Program Files\Wind\Wind.NET.Client\WindNET\x64",
    ],
    "Linux": [],  # Wind is macOS/Windows only
}

# Default EDB indicator codes
DEFAULT_SPREAD_CODES: dict[str, str] = {
    "spread_all": "M0017142",   # 地方债信用利差综合
    "spread_5y": "M0017143",    # 5年期
    "spread_10y": "M0017144",   # 10年期
    "spread_30y": "M0017145",   # 30年期
}

# Credit spread comparison indicators (AAA-rated)
# Uncomment and fill in actual Wind EDB codes when available
CREDIT_SPREAD_CODES: dict[str, str] = {
    # "credit_corp_aaa_3y":  "M00XXXXX",  # 企业债AAA 3Y信用利差
    # "credit_corp_aaa_5y":  "M00XXXXX",  # 企业债AAA 5Y信用利差
    # "credit_corp_aaa_7y":  "M00XXXXX",  # 企业债AAA 7Y信用利差
    # "credit_corp_aaa_10y": "M00XXXXX",  # 企业债AAA 10Y信用利差
    # "credit_mtn_aaa_3y":   "M00XXXXX",  # 中票AAA 3Y信用利差
    # "credit_mtn_aaa_5y":   "M00XXXXX",  # 中票AAA 5Y信用利差
    # "credit_mtn_aaa_7y":   "M00XXXXX",  # 中票AAA 7Y信用利差
    # "credit_mtn_aaa_10y":  "M00XXXXX",  # 中票AAA 10Y信用利差
}


class WindClient:
    """
    Managed Wind Python API client with auto-path detection and retry.

    Parameters
    ----------
    wind_path : Override auto-detected Wind Python path.
    max_retries : Number of retry attempts for transient failures.
    auto_connect : Connect automatically on first fetch call.
    """

    def __init__(
        self,
        wind_path: str | None = None,
        max_retries: int = 2,
        auto_connect: bool = True,
    ) -> None:
        self._w = None  # WindPy module (lazy import)
        self._connected: bool = False
        self._max_retries = max_retries
        self._auto_connect = auto_connect
        self._wind_path = wind_path or self._detect_wind_path()

    def __enter__(self) -> "WindClient":
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        """Check if Wind session is active."""
        if self._w is None:
            return False
        try:
            return self._w.isconnected()
        except Exception:
            return self._connected

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Initialise Wind Python API and start session."""
        if self.is_connected:
            logger.debug("Wind already connected, skipping")
            return

        self._setup_path()
        self._import_windpy()

        logger.info("Connecting to Wind terminal...")
        ret = self._w.start()
        if ret.ErrorCode != 0:
            raise ConnectionError(
                f"Wind start failed: ErrorCode={ret.ErrorCode}. "
                "Ensure Wind terminal is running and license is valid."
            )
        self._connected = True
        logger.info("✓ Wind connected (v%s)", self._w.__version__ if hasattr(self._w, '__version__') else "?")

    def disconnect(self) -> None:
        """Stop Wind session gracefully."""
        if self._w is not None and self._connected:
            try:
                self._w.stop()
                logger.info("✓ Wind disconnected")
            except Exception as exc:
                logger.warning("Wind disconnect error (non-fatal): %s", exc)
            finally:
                self._connected = False

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_edb(
        self,
        codes: dict[str, str] | list[str],
        start_date: str,
        end_date: str | None = None,
        fill_method: str = "Previous",
    ) -> pd.DataFrame:
        """
        Fetch EDB (Economic Data Browser) indicators from Wind.

        Parameters
        ----------
        codes : Either a dict {column_name: edb_code} or a list of edb_codes.
        start_date : Start date string "YYYY-MM-DD".
        end_date : End date string. Defaults to today.
        fill_method : Wind fill option — "Previous" (forward-fill), "None", etc.

        Returns
        -------
        DataFrame with 'date' column + one column per indicator.
        """
        if self._auto_connect and not self.is_connected:
            self.connect()

        if end_date is None:
            end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

        # Normalise codes to {col_name: edb_code}
        if isinstance(codes, list):
            code_map = {code: code for code in codes}
        else:
            code_map = dict(codes)

        logger.info(
            "Fetching %d EDB indicators from %s to %s (fill=%s)",
            len(code_map), start_date, end_date, fill_method,
        )

        frames: list[pd.Series] = []
        failed: list[str] = []

        for col_name, edb_code in code_map.items():
            series = self._fetch_single_edb(edb_code, col_name, start_date, end_date, fill_method)
            if series is not None:
                frames.append(series)
            else:
                failed.append(f"{col_name}({edb_code})")

        if not frames:
            raise RuntimeError(f"All {len(code_map)} EDB fetches failed: {failed}")

        if failed:
            logger.warning("Partial success: %d/%d indicators fetched. Failed: %s",
                           len(frames), len(code_map), failed)

        df = pd.concat(frames, axis=1).reset_index()
        df = df.rename(columns={"index": "date"})

        logger.info("✓ EDB data loaded: %d rows × %d columns", len(df), len(df.columns))
        return df

    def fetch_edb_raw(
        self,
        edb_code: str,
        start_date: str,
        end_date: str | None = None,
    ) -> tuple[list, list]:
        """
        Fetch a single EDB code and return raw (times, data) lists.
        For advanced users who need the raw Wind response.
        """
        if self._auto_connect and not self.is_connected:
            self.connect()

        if end_date is None:
            end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

        raw = self._w.edb(edb_code, start_date, end_date)
        if raw.ErrorCode != 0:
            raise RuntimeError(f"Wind EDB error for {edb_code}: code={raw.ErrorCode}")
        return raw.Times, raw.Data[0]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_single_edb(
        self,
        edb_code: str,
        col_name: str,
        start_date: str,
        end_date: str,
        fill_method: str,
    ) -> pd.Series | None:
        """Fetch a single EDB indicator with retry."""
        for attempt in range(1, self._max_retries + 1):
            try:
                raw = self._w.edb(edb_code, start_date, end_date, f"Fill={fill_method}")

                if raw.ErrorCode != 0:
                    logger.warning(
                        "Wind EDB %s (attempt %d/%d): ErrorCode=%d",
                        edb_code, attempt, self._max_retries, raw.ErrorCode,
                    )
                    if attempt < self._max_retries:
                        import time
                        time.sleep(1.0 * attempt)
                        continue
                    return None

                data = raw.Data[0]
                times = pd.to_datetime(raw.Times)

                if len(data) == 0:
                    logger.warning("Wind EDB %s returned empty data", edb_code)
                    return None

                series = pd.Series(data, index=times, name=col_name, dtype=np.float64)
                # Replace Wind's NaN sentinel values
                series = series.replace({-999.0: np.nan, 999.0: np.nan, -9999.0: np.nan})
                logger.info("  ✓ %s (%s): %d observations", col_name, edb_code, len(series))
                return series

            except Exception as exc:
                logger.warning(
                    "Wind EDB %s (attempt %d/%d) exception: %s",
                    edb_code, attempt, self._max_retries, exc,
                )
                if attempt < self._max_retries:
                    import time
                    time.sleep(1.0 * attempt)

        return None

    @staticmethod
    def _detect_wind_path() -> str | None:
        """Auto-detect Wind Python API path based on platform."""
        system = platform.system()
        candidates = _WIND_PATHS.get(system, [])

        for path_str in candidates:
            p = Path(path_str)
            if p.exists():
                logger.info("Auto-detected Wind path: %s", p)
                return str(p)

        logger.debug("No Wind path auto-detected on %s", system)
        return None

    def _setup_path(self) -> None:
        """Add Wind Python path to sys.path if needed."""
        if self._wind_path and self._wind_path not in sys.path:
            sys.path.insert(0, self._wind_path)
            logger.debug("Added Wind path to sys.path: %s", self._wind_path)

    def _import_windpy(self) -> None:
        """Import WindPy module with informative error."""
        try:
            from WindPy import w  # type: ignore[import-not-found]
            self._w = w
        except ImportError:
            path_hint = (
                f" (searched: {self._wind_path})" if self._wind_path
                else " (no Wind path detected)"
            )
            raise ImportError(
                "Wind Python API (WindPy) not found. "
                "Install Wind terminal and ensure the Python package is on sys.path. "
                f"macOS default: /Applications/Wind API.app/Contents/python{path_hint}"
            ) from None
