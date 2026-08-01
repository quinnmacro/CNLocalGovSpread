"""
Bayesian Structural Time Series signal extractor using PyMC + ADVI.

Bayesian Local Level Model:
    Observation:  y_t = level_t + eps_t,   eps_t ~ N(0, sigma_obs^2)
    Level:        level_t = level_{t-1} + eta_t,  eta_t ~ N(0, sigma_level^2)

Uses PyMC with Automatic Differentiation Variational Inference (ADVI)
for fast approximate Bayesian inference. Returns posterior mean of the
latent level as the "signal" plus 95% credible intervals.

Key advantage over Kalman filter: full posterior uncertainty quantification
via credible intervals, not just point estimates.
"""

from __future__ import annotations

import time
from typing import Optional, Self

import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit  # sigmoid function

from src.core.base import NotFittedError, SignalExtractor
from src.core.logging_config import get_logger
from src.core.types import BayesianSTSResult

logger = get_logger(__name__)


class BayesianSTSSignalExtractor(SignalExtractor):
    """
    Bayesian Local Level Model via PyMC + ADVI for signal extraction.

    Uses variational inference (ADVI) for fast approximate Bayesian inference.
    The latent level is modelled as a Gaussian random walk; observation noise
    is Gaussian. Posterior mean of the level serves as the extracted signal.

    Parameters
    ----------
    n_advi_steps :
        Number of ADVI optimisation steps. More = better approximation
        but slower. Default 15000.
    n_samples :
        Number of posterior samples to draw from the fitted approximation.
        Default 300.
    rolling_window :
        Window (days) for rolling std of deviation. Default 60.
    random_seed :
        Random seed for reproducibility. Default 42.
    """

    def __init__(
        self,
        n_advi_steps: int = 15000,
        n_samples: int = 300,
        rolling_window: int = 60,
        random_seed: int = 42,
    ) -> None:
        if rolling_window < 5:
            raise ValueError(f"rolling_window must be >= 5, got {rolling_window}")
        if n_advi_steps < 100:
            raise ValueError(f"n_advi_steps must be >= 100, got {n_advi_steps}")
        if n_samples < 10:
            raise ValueError(f"n_samples must be >= 10, got {n_samples}")

        self.n_advi_steps = n_advi_steps
        self.n_samples = n_samples
        self.rolling_window = rolling_window
        self.random_seed = random_seed

        # Caching internals
        self._spread_hash: Optional[int] = None
        self._spread: Optional[pd.Series] = None

        logger.debug(
            "BayesianSTSSignalExtractor: advi_steps=%d, samples=%d, window=%d",
            n_advi_steps, n_samples, rolling_window,
        )

    # ------------------------------------------------------------------
    # SignalExtractor interface
    # ------------------------------------------------------------------

    def fit(self, spread: pd.Series) -> Self:
        """
        Build and fit the Bayesian Local Level model on the spread series.

        Uses result caching: if the data hasn't changed (same hash),
        returns the cached result without re-fitting.
        """
        if len(spread) < 30:
            raise ValueError(
                f"Bayesian STS requires at least 30 observations, got {len(spread)}"
            )

        # --- Caching check ---
        spread_hash = hash(spread.values.tobytes())
        if self._result is not None and self._spread_hash == spread_hash:
            logger.info("Returning cached Bayesian STS result (data unchanged).")
            return self

        self._spread = spread.copy()
        self._spread_hash = spread_hash
        n = len(spread)

        y = spread.values.astype(np.float64)
        y_std = float(np.std(y))
        y_mean = float(np.mean(y))

        logger.info(
            "Fitting Bayesian Local Level model on %d observations "
            "(ADVI steps=%d, samples=%d)...",
            n, self.n_advi_steps, self.n_samples,
        )

        t_start = time.time()

        # --- Build PyMC model ---
        with pm.Model() as model:
            # Priors on noise scales
            sigma_level = pm.HalfNormal("sigma_level", sigma=y_std * 0.05)
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=y_std * 0.5)

            # Latent level: Gaussian random walk
            level = pm.GaussianRandomWalk(
                "level",
                sigma=sigma_level,
                shape=n,
                init_dist=pm.Normal.dist(mu=y_mean, sigma=y_std * 0.5),
            )

            # Observation model
            pm.Normal("obs", mu=level, sigma=sigma_obs, observed=y)

            # --- ADVI fitting ---
            approx = pm.fit(
                n=self.n_advi_steps,
                method="advi",
                progressbar=False,
                random_seed=self.random_seed,
            )
            trace = approx.sample(self.n_samples, random_seed=self.random_seed)

        fitting_time = time.time() - t_start

        # --- Extract posterior statistics ---
        # trace.posterior["level"] has shape (chains, draws, n)
        level_posterior = trace.posterior["level"].values  # (chains, draws, n)

        # Flatten chains and draws -> (total_samples, n)
        n_chains, n_draws, _ = level_posterior.shape
        level_samples = level_posterior.reshape(-1, n)  # (total_samples, n)

        # Posterior mean (signal)
        signal_mean = level_samples.mean(axis=0)

        # 95% credible intervals (2.5th and 97.5th percentiles)
        signal_lower = np.percentile(level_samples, 2.5, axis=0)
        signal_upper = np.percentile(level_samples, 97.5, axis=0)

        # Average CI width
        ci_width_mean = float(np.mean(signal_upper - signal_lower))

        # Extract hyper-parameter posteriors
        sigma_level_posterior = trace.posterior["sigma_level"].values.flatten()
        sigma_obs_posterior = trace.posterior["sigma_obs"].values.flatten()
        sigma_level_mean = float(np.mean(sigma_level_posterior))
        sigma_obs_mean = float(np.mean(sigma_obs_posterior))

        logger.info(
            "Bayesian STS fit complete in %.1fs: sigma_level=%.6f, sigma_obs=%.6f, CI_width=%.4f",
            fitting_time, sigma_level_mean, sigma_obs_mean, ci_width_mean,
        )

        # --- Build result series ---
        signal = pd.Series(signal_mean, index=spread.index, name="signal")
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

        signal_lower_s = pd.Series(signal_lower, index=spread.index, name="signal_lower")
        signal_upper_s = pd.Series(signal_upper, index=spread.index, name="signal_upper")

        self._result = BayesianSTSResult(
            method="Bayesian(LocalLevel+ADVI)",
            signal=signal,
            trend=trend,
            deviation=deviation,
            deviation_zscore=deviation_zscore,
            signal_strength=signal_strength,
            is_overvalued=is_overvalued,
            is_undervalued=is_undervalued,
            # Bayesian-specific
            signal_lower=signal_lower_s,
            signal_upper=signal_upper_s,
            ci_width_mean=ci_width_mean,
            sigma_level_mean=sigma_level_mean,
            sigma_obs_mean=sigma_obs_mean,
            n_samples=n_chains * n_draws,
            fitting_time_sec=fitting_time,
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
