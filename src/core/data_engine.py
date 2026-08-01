"""
Unified data loading with source abstraction (MOCK / CSV / Wind).

DataEngine.load() dispatches to the appropriate loader based on DataConfig.source.
All returned DataFrames share the same schema:
    columns = [date, spread_all, spread_5y, spread_10y, spread_30y]
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.core.config import DataSource, DataConfig, get_settings
from src.core.logging_config import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Wind EDB indicator codes for China local gov spreads
_WIND_EDB_CODES: dict[str, str] = {
    "spread_all": "M0017142",
    "spread_5y": "M0017143",
    "spread_10y": "M0017144",
    "spread_30y": "M0017145",
}

REQUIRED_COLUMNS = ["date", "spread_all", "spread_5y", "spread_10y", "spread_30y"]


class DataEngine:
    """Unified data loading with source abstraction (MOCK / CSV / Wind)."""

    def __init__(self, config: DataConfig | None = None) -> None:
        self._config = config or get_settings().data
        logger.info("DataEngine initialised with source=%s", self._config.source.value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Load raw data from the configured source and return a cleaned DataFrame."""
        logger.info("Loading data from source=%s", self._config.source.value)

        if self._config.source == DataSource.MOCK:
            df = self._generate_mock_data()
        elif self._config.source == DataSource.CSV:
            df = self._load_csv()
        elif self._config.source == DataSource.WIND:
            df = self._load_wind()
        else:
            raise ValueError(f"Unsupported data source: {self._config.source}")

        df = self._validate_columns(df)
        logger.info(
            "Loaded %d rows, %d columns; date range [%s, %s]",
            len(df), len(df.columns),
            df["date"].iloc[0], df["date"].iloc[-1],
        )
        return self.clean(df)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame with MAD-based outlier detection and forward-fill.

        Steps:
        1. Sort by date and drop duplicates.
        2. Forward-fill small gaps (up to 3 days).
        3. Flag outliers via Median Absolute Deviation (MAD).
        4. Replace outliers with NaN then forward-fill again.
        """
        out = df.copy()
        out = out.sort_values("date").reset_index(drop=True)
        out = out.drop_duplicates(subset=["date"], keep="last")

        spread_cols = [c for c in out.columns if c.startswith("spread_")]

        out[spread_cols] = out[spread_cols].ffill(limit=3)

        threshold = self._config.mad_threshold
        outlier_count = 0
        for col in spread_cols:
            series = out[col].dropna()
            if series.empty:
                continue
            median = series.median()
            mad = np.median(np.abs(series - median)) / 0.6745
            if mad < 1e-10:
                continue
            mask = np.abs(out[col] - median) > threshold * mad
            n_flagged = mask.sum()
            if n_flagged > 0:
                logger.warning(
                    "MAD outlier detection on '%s': flagged %d points (threshold=%.1f)",
                    col, n_flagged, threshold,
                )
                out.loc[mask, col] = np.nan
                outlier_count += n_flagged

        out[spread_cols] = out[spread_cols].ffill()
        out[spread_cols] = out[spread_cols].bfill()

        if outlier_count > 0:
            logger.info("Total outliers replaced and re-filled: %d", outlier_count)
        return out

    def compute_returns(self, df: pd.DataFrame, column: str = "spread_all") -> pd.Series:
        """
        Compute daily first-differences of spread levels.

        These are **spread changes in bps**, not percentage returns.
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found. Available: {list(df.columns)}")
        diff = df[column].diff().dropna()
        diff.name = f"d_{column}"
        return diff

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _generate_mock_data(self, n: int = 1500) -> pd.DataFrame:
        """
        Generate synthetic spread data using AR(1) + GARCH(1,1) DGP with Student-t errors.
        """
        from src.core.simulator import SpreadSimulator

        logger.info("Generating %d rows of mock AR(1)+GARCH data", n)

        rng = np.random.default_rng(seed=42)

        sim = SpreadSimulator(
            phi=0.95, omega=0.02, alpha=0.08, beta=0.88, df_t=5.0
        )
        spreads_all, _ = sim.simulate_single_path(n_steps=n, seed=42)

        base_level = 30.0
        spread_all = base_level + np.cumsum(spreads_all * 0.1)

        noise_scale = 0.5
        spread_5y = spread_all + rng.normal(0, noise_scale, n) + 2.0
        spread_10y = spread_all + rng.normal(0, noise_scale * 1.5, n) + 5.0
        spread_30y = spread_all + rng.normal(0, noise_scale * 2.0, n) + 10.0

        dates = pd.bdate_range(start="2018-01-02", periods=n, freq="B")
        df = pd.DataFrame({
            "date": dates,
            "spread_all": spread_all,
            "spread_5y": spread_5y,
            "spread_10y": spread_10y,
            "spread_30y": spread_30y,
        })
        logger.info("Mock data generated: %d rows", len(df))
        return df

    def _load_csv(self) -> pd.DataFrame:
        """Load data from CSV file specified in config."""
        path = Path(self._config.csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV data file not found: {path}")

        logger.info("Loading CSV from %s", path)
        df = pd.read_csv(path, parse_dates=["date"], dayfirst=False)

        if self._config.start_date:
            start = pd.Timestamp(self._config.start_date)
            df = df[df["date"] >= start].reset_index(drop=True)
            logger.info(
                "Filtered to start_date=%s (%d rows remaining)",
                self._config.start_date, len(df),
            )

        return df

    def _load_wind(self) -> pd.DataFrame:
        """
        Load data from Wind Financial Terminal via WindPy.
        Uses w.edb() to fetch EDB indicators M0017142–M0017145.
        """
        try:
            from WindPy import w  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "WindPy is not installed. Install with: pip install WindPy  "
                "or change DataConfig.source to 'mock' or 'csv'."
            ) from exc

        if not w.isconnected():
            logger.info("Connecting to Wind terminal...")
            w.start()

        start_date = self._config.start_date
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

        logger.info("Fetching Wind EDB data: %s to %s", start_date, end_date)

        frames = []
        for col_name, edb_code in _WIND_EDB_CODES.items():
            raw = w.edb(edb_code, start_date, end_date, "Fill=Previous")
            if raw.ErrorCode != 0:
                raise RuntimeError(
                    f"Wind EDB error for {edb_code}: code={raw.ErrorCode}, msg={raw.Data}"
                )
            series = pd.Series(raw.Data[0], index=pd.to_datetime(raw.Times), name=col_name)
            frames.append(series)

        df = pd.concat(frames, axis=1).reset_index()
        df = df.rename(columns={"index": "date"})
        logger.info("Wind data loaded: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required columns are present."""
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame missing required columns: {missing}. "
                f"Got: {list(df.columns)}"
            )
        return df
