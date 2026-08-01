"""
Extreme Value Theory (EVT) risk analysis via Peaks Over Threshold (POT).

Fits a Generalized Pareto Distribution (GPD) to tail exceedances and
computes EVT-VaR and EVT-ES.

Reference: McNeil, Frey, Embrechts (2005), Quantitative Risk Management.
Implements the RiskAnalyzer ABC.
"""

from __future__ import annotations

from typing import Optional, Self

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.core.base import RiskAnalyzer
from src.core.config import get_settings
from src.core.logging_config import get_logger
from src.core.types import BacktestResult, RiskResult

logger = get_logger(__name__)


class EVTAnalyzer(RiskAnalyzer):
    """
    EVT-based risk analyzer using Peaks Over Threshold (POT).

    Fits GPD to the right tail (spread widening risk).
    """

    def __init__(
        self,
        threshold_percentile: float | None = None,
    ) -> None:
        cfg = get_settings().risk
        self._threshold_pct = threshold_percentile or cfg.evt_threshold_percentile
        self._threshold: float = 0.0
        self._gpd_shape: float = 0.0
        self._gpd_scale: float = 0.0
        self._n_exceedances: int = 0
        self._n_total: int = 0
        self._returns: Optional[pd.Series] = None

    def fit(self, returns: pd.Series, confidence: float = 0.99) -> Self:
        """
        Fit GPD to tail exceedances and compute EVT-VaR/ES.

        Parameters
        ----------
        returns : pd.Series
            Spread changes (bps). We model the RIGHT tail (large positive = spread widening).
        confidence : float
            VaR/ES confidence level (e.g. 0.99).
        """
        self._returns = returns
        r = returns.dropna().values.astype(float)
        self._n_total = len(r)

        # Threshold
        self._threshold = float(np.quantile(r, self._threshold_pct))
        exceedances = r[r > self._threshold] - self._threshold
        self._n_exceedances = len(exceedances)

        if self._n_exceedances < 10:
            logger.warning(
                "Only %d exceedances (threshold=%.2f). EVT estimates may be unreliable.",
                self._n_exceedances, self._threshold,
            )

        if self._n_exceedances == 0:
            # Fallback to empirical quantile
            var_val = float(np.quantile(r, confidence))
            es_val = float(r[r > var_val].mean()) if np.any(r > var_val) else var_val
            self._result = RiskResult(
                method="empirical", confidence=confidence,
                var=var_val, es=es_val, n_exceedances=0,
            )
            return self

        # Fit GPD (MLE)
        try:
            shape, loc, scale = sp_stats.genpareto.fit(exceedances, floc=0)
            self._gpd_shape = float(shape)
            self._gpd_scale = float(scale)
        except Exception as exc:
            logger.warning("GPD fitting failed: %s. Using empirical fallback.", exc)
            var_val = float(np.quantile(r, confidence))
            es_val = float(r[r > var_val].mean()) if np.any(r > var_val) else var_val
            self._result = RiskResult(
                method="empirical", confidence=confidence,
                var=var_val, es=es_val, n_exceedances=self._n_exceedances,
            )
            return self

        # Safety check on shape parameter
        if abs(self._gpd_shape) > 1.0:
            logger.warning("GPD shape ξ=%.4f is extreme, using empirical fallback.", self._gpd_shape)
            var_val = float(np.quantile(r, confidence))
            es_val = float(r[r > var_val].mean()) if np.any(r > var_val) else var_val
            self._result = RiskResult(
                method="empirical", confidence=confidence,
                var=var_val, es=es_val,
                gpd_shape=self._gpd_shape, gpd_scale=self._gpd_scale,
                n_exceedances=self._n_exceedances, threshold=self._threshold,
            )
            return self

        # EVT-VaR (POT method)
        # VaR_α = u + (σ/ξ) * [((n/Nu)*(1-α))^(-ξ) - 1]
        n = self._n_total
        nu = self._n_exceedances
        xi = self._gpd_shape
        sigma = self._gpd_scale
        u = self._threshold

        if abs(xi) < 1e-6:
            # Exponential limit
            var_val = u - sigma * np.log((n / nu) * (1 - confidence))
        else:
            power = ((n / nu) * (1 - confidence)) ** (-xi)
            var_val = u + (sigma / xi) * (power - 1)

        # EVT-ES
        # ES_α = VaR/(1-ξ) + (σ - ξ*u)/(1-ξ)   for ξ < 1
        if abs(xi) < 1e-6:
            es_val = var_val + sigma
        elif xi < 1:
            es_val = var_val / (1 - xi) + (sigma - xi * u) / (1 - xi)
        else:
            # ES undefined for xi >= 1 (infinite mean)
            es_val = float("inf")

        # Tail index (Hill-type: 1/xi)
        tail_index = 1.0 / xi if abs(xi) > 1e-6 else float("inf")

        self._result = RiskResult(
            method="evt_pot",
            confidence=confidence,
            var=float(var_val),
            es=float(es_val),
            tail_index=float(tail_index),
            threshold=u,
            gpd_shape=xi,
            gpd_scale=sigma,
            n_exceedances=nu,
        )

        logger.info(
            "EVT fitted: ξ=%.4f, σ=%.4f, VaR=%.4f, ES=%.4f (n_exceed=%d)",
            xi, sigma, var_val, es_val, nu,
        )
        return self

    def backtest(self, returns: pd.Series, var_series: pd.Series) -> BacktestResult:
        """Delegates to VaRBacktest."""
        from src.risk.backtest import VaRBacktest
        bt = VaRBacktest()
        return bt.full_backtest(returns, var_series, confidence=self._result.confidence)

    def hill_estimator(self, k_percentile: float = 0.10) -> dict:
        """
        Non-parametric Hill tail index estimator.

        Parameters
        ----------
        k_percentile : float
            Fraction of top observations to use (default 10%).

        Returns
        -------
        dict with tail_index, shape, threshold, k.
        """
        if self._returns is None:
            raise ValueError("Call fit() first.")

        r = self._returns.dropna().values
        sorted_r = np.sort(r)[::-1]  # descending
        k = max(10, int(len(sorted_r) * k_percentile))
        k = min(k, len(sorted_r) - 1)

        threshold = sorted_r[k]
        exceedances = sorted_r[:k]

        safe = exceedances[exceedances > 0]
        if len(safe) == 0 or threshold <= 0:
            return {"tail_index": float("inf"), "shape": 0.0, "threshold": threshold, "k": k}

        xi_hill = float(np.mean(np.log(safe / threshold)))
        tail_index = 1.0 / xi_hill if abs(xi_hill) > 1e-6 else float("inf")

        return {
            "tail_index": tail_index,
            "shape": xi_hill,
            "threshold": float(threshold),
            "k": k,
        }

    def mean_excess_data(self, n_thresholds: int = 50) -> pd.DataFrame:
        """
        Compute mean excess plot data for threshold selection.

        Returns DataFrame with columns: threshold, mean_excess, n_exceedances.
        """
        if self._returns is None:
            raise ValueError("Call fit() first.")

        r = self._returns.dropna().values
        r_min, r_max = np.quantile(r, 0.90), np.quantile(r, 0.999)
        thresholds = np.linspace(r_min, r_max, n_thresholds)

        rows = []
        for u in thresholds:
            exc = r[r > u] - u
            if len(exc) >= 5:
                rows.append({
                    "threshold": u,
                    "mean_excess": float(np.mean(exc)),
                    "n_exceedances": len(exc),
                })

        return pd.DataFrame(rows)
