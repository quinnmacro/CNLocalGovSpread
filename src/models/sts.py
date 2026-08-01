"""
Structural Time Series signal extractor using UnobservedComponents.

Local Linear Trend Model (state-space form):
    Observation:  y_t = μ_t + ε_t,         ε_t ~ N(0, σ²_irregular)
    Level:        μ_{t+1} = μ_t + ν_t + η_t, η_t ~ N(0, σ²_level)
    Slope:        ν_{t+1} = ν_t + ζ_t,     ζ_t ~ N(0, σ²_trend)

The smoothed level μ_t is extracted as the "signal" (fundamental trend).
The slope ν_t captures drift / speed of trend change.
Deviation = y_t − μ_t measures how far the spread is from its trend.
Z-score = deviation / rolling_std(deviation) normalises for comparison.
Signal strength uses a sigmoid mapping: S = 1 / (1 + exp(−|z|))

Uses statsmodels UnobservedComponents with level='local linear trend'
(best AIC in testing among structural specifications).
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid function
from statsmodels.tsa.statespace.structural import UnobservedComponents  # type: ignore[import-untyped]

from src.core.base import NotFittedError, SignalExtractor
from src.core.logging_config import get_logger
from src.core.types import STSResult

logger = get_logger(__name__)


class STSSignalExtractor(SignalExtractor):
    """
    Structural Time Series signal extractor using local linear trend.

    Parameters
    ----------
    rolling_window :
        Window (days) for rolling std of deviation. Default 60.
    """

    def __init__(self, rolling_window: int = 60) -> None:
        if rolling_window < 5:
            raise ValueError(f"rolling_window must be >= 5, got {rolling_window}")

        self.rolling_window = rolling_window

        self._spread: pd.Series | None = None
        self._fitted_model = None

        logger.debug("STSSignalExtractor: window=%d", rolling_window)

    # ------------------------------------------------------------------
    # SignalExtractor interface
    # ------------------------------------------------------------------

    def fit(self, spread: pd.Series) -> Self:
        """
        Extract signal from a spread level series using a local linear trend STS model.
        """
        if len(spread) < 30:
            raise ValueError(
                f"STS model requires at least 30 observations, got {len(spread)}"
            )

        self._spread = spread.copy()
        y = spread.values.astype(np.float64)

        model = UnobservedComponents(
            y,
            level="local linear trend",
        )

        logger.info(
            "Fitting STS (local linear trend) on %d observations...", len(y)
        )

        try:
            fit_result = model.fit(disp=False, maxiter=300)
            self._fitted_model = fit_result
        except Exception as exc:
            logger.error("STS fitting failed: %s", exc)
            raise

        # Extract smoothed components
        smoothed = fit_result.smoothed_state
        # smoothed_state shape: (k_states, n_obs)
        # For local linear trend: state 0 = level, state 1 = slope
        level_arr = smoothed[0]
        slope_arr = smoothed[1] if smoothed.shape[0] > 1 else np.zeros_like(level_arr)

        level = pd.Series(level_arr, index=spread.index, name="level")
        slope = pd.Series(slope_arr, index=spread.index, name="slope")

        # Signal = smoothed level (the fundamental trend)
        signal = level.copy()
        signal.name = "signal"

        # Trend = slope (drift / speed of trend change)
        trend = slope.copy()
        trend.name = "trend"

        # Deviation from signal
        deviation = spread - signal
        deviation.name = "deviation"

        # Z-score normalisation
        rolling_std = deviation.rolling(window=self.rolling_window, min_periods=10).std()
        rolling_std = rolling_std.replace(0, np.nan).ffill().bfill()
        rolling_std = rolling_std.clip(lower=1e-8)

        deviation_zscore = deviation / rolling_std
        deviation_zscore.name = "deviation_zscore"

        # Signal strength via sigmoid
        z_last = abs(float(deviation_zscore.iloc[-1]))
        signal_strength = float(expit(z_last))

        is_overvalued = float(deviation_zscore.iloc[-1]) > 1.5
        is_undervalued = float(deviation_zscore.iloc[-1]) < -1.5

        # Irregular component (observation noise residuals)
        # resid is y_t - filtered/signal level; use smoothed resid as irregular
        irregular = pd.Series(
            fit_result.resid, index=spread.index, name="irregular"
        )

        # Extract variance parameters
        # UnobservedComponents param names for local linear trend:
        #   sigma2.irregular, sigma2.level, sigma2.trend
        param_dict = dict(zip(fit_result.param_names, fit_result.params))
        sigma2_irregular = float(param_dict.get("sigma2.irregular", 0.0))
        sigma2_level = float(param_dict.get("sigma2.level", 0.0))
        sigma2_trend = float(param_dict.get("sigma2.trend", 0.0))

        logger.info(
            "STS fit: σ²_level=%.6f, σ²_trend=%.6f, σ²_irregular=%.6f, AIC=%.2f",
            sigma2_level,
            sigma2_trend,
            sigma2_irregular,
            fit_result.aic,
        )

        self._result = STSResult(
            method="STS(LocalLinearTrend)",
            signal=signal,
            trend=trend,
            deviation=deviation,
            deviation_zscore=deviation_zscore,
            signal_strength=signal_strength,
            is_overvalued=is_overvalued,
            is_undervalued=is_undervalued,
            level=level,
            slope=slope,
            irregular=irregular,
            aic=float(fit_result.aic),
            bic=float(fit_result.bic),
            n_params=int(fit_result.params.shape[0]) if hasattr(fit_result.params, "shape") else len(fit_result.params),
            sigma2_level=sigma2_level,
            sigma2_trend=sigma2_trend,
            sigma2_irregular=sigma2_irregular,
        )

        logger.info(
            "Signal extraction complete: strength=%.4f, z=%.2f, %s",
            signal_strength,
            float(deviation_zscore.iloc[-1]),
            "overvalued"
            if is_overvalued
            else ("undervalued" if is_undervalued else "fair"),
        )
        return self

    def get_signal_deviation(self) -> pd.Series:
        """Return deviation of current spread from extracted signal."""
        if self._result is None:
            raise NotFittedError("Call fit() before accessing signal deviation.")
        return self._result.deviation
