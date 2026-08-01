"""
Kalman filter signal extractor using the Local Level Model.

Local Level Model (state-space form):
    Observation:  y_t = μ_t + ε_t,     ε_t ~ N(0, σ²_ε)
    Transition:   μ_{t+1} = μ_t + η_t, η_t ~ N(0, σ²_η)

The smoothed state μ_t is extracted as the "signal" (fundamental trend).
Deviation = y_t − μ_t measures how far the spread is from its trend.
Z-score = deviation / rolling_std(deviation) normalises this for comparison.
Signal strength uses a sigmoid mapping: S = 1 / (1 + exp(−|z|))

Implementation:
- Uses statsmodels MLEModel for Kalman filter (statespace representation)
- Fits by MLE to estimate σ²_η and σ²_ε
- Signal = smoothed state estimates (backward pass)
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid function
from statsmodels.tsa.statespace.mlemodel import MLEModel  # type: ignore[import-untyped]

from src.core.base import NotFittedError, SignalExtractor
from src.core.logging_config import get_logger
from src.core.types import SignalResult

logger = get_logger(__name__)


class _LocalLevelModel(MLEModel):
    """
    statsmodels MLEModel subclass implementing the Local Level Model.

    Parameters estimated by MLE:
    - sigma2_eta: transition (state) variance
    - sigma2_eps: observation (measurement) variance
    """

    def __init__(self, endog: np.ndarray) -> None:
        super().__init__(
            endog=endog,
            k_states=1,
            k_posdef=1,
            initialization="approximate_diffuse",
        )
        self["design", 0, 0] = 1.0
        self["transition", 0, 0] = 1.0
        self["selection", 0, 0] = 1.0

    @property
    def param_names(self) -> list[str]:
        return ["sigma2_eta", "sigma2_eps"]

    @property
    def start_params(self) -> np.ndarray:
        y = self.endog[:, 0] if self.endog.ndim > 1 else self.endog
        var_y = np.var(y)
        return np.array([var_y * 0.01, var_y * 0.99])

    def transform_params(self, unconstrained: np.ndarray) -> np.ndarray:
        """Ensure variances are positive."""
        return unconstrained ** 2

    def untransform_params(self, constrained: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(constrained, 1e-12))

    def update(self, params: np.ndarray, **kwargs) -> None:  # type: ignore[override]
        params = super().update(params, **kwargs)
        self["state_cov", 0, 0] = params[0]
        self["obs_cov", 0, 0] = params[1]


class KalmanSignalExtractor(SignalExtractor):
    """
    Local Level Model via Kalman filter for trend/signal extraction.

    Parameters
    ----------
    transition_variance :
        Initial guess for state transition variance σ²_η.
        Lower values → smoother signal.
    observation_variance :
        Initial guess for observation noise variance σ²_ε.
    rolling_window :
        Window (days) for rolling std of deviation. Default 60.
    """

    def __init__(
        self,
        transition_variance: float = 0.01,
        observation_variance: float = 1.0,
        rolling_window: int = 60,
    ) -> None:
        if rolling_window < 5:
            raise ValueError(f"rolling_window must be >= 5, got {rolling_window}")

        self.transition_variance = transition_variance
        self.observation_variance = observation_variance
        self.rolling_window = rolling_window

        self._spread: pd.Series | None = None
        self._fitted_model = None

        logger.debug(
            "KalmanSignalExtractor: σ²_η=%.4f, σ²_ε=%.4f, window=%d",
            transition_variance, observation_variance, rolling_window,
        )

    # ------------------------------------------------------------------
    # SignalExtractor interface
    # ------------------------------------------------------------------

    def fit(self, spread: pd.Series) -> Self:
        """
        Extract signal from a spread level series using the Local Level Model.
        """
        if len(spread) < 30:
            raise ValueError(f"Kalman filter requires at least 30 observations, got {len(spread)}")

        self._spread = spread.copy()
        y = spread.values.astype(np.float64)

        model = _LocalLevelModel(y)

        logger.info("Fitting Local Level Model via Kalman filter on %d observations...", len(y))

        try:
            fit_result = model.fit(disp=False, maxiter=300)
            self._fitted_model = fit_result
        except Exception as exc:
            logger.error("Kalman filter fitting failed: %s", exc)
            raise

        sigma2_eta = float(fit_result.params[0])
        sigma2_eps = float(fit_result.params[1])
        logger.info(
            "Kalman fit: σ²_η=%.6f, σ²_ε=%.6f, ratio=%.4f",
            sigma2_eta, sigma2_eps,
            sigma2_eta / max(sigma2_eps, 1e-12),
        )

        # Extract smoothed state (= signal)
        smoothed = fit_result.smoothed_state[0]
        signal = pd.Series(smoothed, index=spread.index, name="signal")

        trend = signal.copy()
        trend.name = "trend"

        deviation = spread - signal
        deviation.name = "deviation"

        rolling_std = deviation.rolling(window=self.rolling_window, min_periods=10).std()
        rolling_std = rolling_std.replace(0, np.nan).ffill().bfill()
        rolling_std = rolling_std.clip(lower=1e-8)

        deviation_zscore = deviation / rolling_std
        deviation_zscore.name = "deviation_zscore"

        z_last = abs(float(deviation_zscore.iloc[-1]))
        signal_strength = float(expit(z_last))

        is_overvalued = float(deviation_zscore.iloc[-1]) > 1.5
        is_undervalued = float(deviation_zscore.iloc[-1]) < -1.5

        self._result = SignalResult(
            method="Kalman(LocalLevel)",
            signal=signal,
            trend=trend,
            deviation=deviation,
            deviation_zscore=deviation_zscore,
            signal_strength=signal_strength,
            is_overvalued=is_overvalued,
            is_undervalued=is_undervalued,
        )

        logger.info(
            "Signal extraction complete: strength=%.4f, z=%.2f, %s",
            signal_strength,
            float(deviation_zscore.iloc[-1]),
            "overvalued" if is_overvalued else ("undervalued" if is_undervalued else "fair"),
        )
        return self

    def get_signal_deviation(self) -> pd.Series:
        """Return deviation of current spread from extracted signal."""
        if self._result is None:
            raise NotFittedError("Call fit() before accessing signal deviation.")
        return self._result.deviation
