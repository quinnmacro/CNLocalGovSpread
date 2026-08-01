"""
VaR backtesting: Kupiec unconditional and Christoffersen conditional coverage tests.

Verifies that VaR estimates are well-calibrated by comparing
observed violation frequency to expected.
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


class VaRBacktest:
    """
    VaR backtesting engine with Kupiec and Christoffersen tests.
    """

    def kupiec_test(
        self,
        n_violations: int,
        n_obs: int,
        confidence: float,
    ) -> tuple[float, float]:
        """
        Kupiec (1995) unconditional coverage test.

        H0: actual violation rate = expected rate (1 - confidence).
        LR ~ chi²(1) under H0.

        Returns
        -------
        (statistic, pvalue)
        """
        p0 = 1 - confidence  # expected violation rate
        p_hat = n_violations / n_obs if n_obs > 0 else 0

        if n_violations == 0:
            # LR = -2 * log((1-p0)^n / 1) = -2 * n * log(1-p0)
            lr = -2 * n_obs * np.log(1 - p0)
        elif n_violations == n_obs:
            lr = 1e6  # degenerate
        else:
            # LR = -2 * [log(p0^v * (1-p0)^(n-v)) - log(p_hat^v * (1-p_hat)^(n-v))]
            log_h0 = n_violations * np.log(p0) + (n_obs - n_violations) * np.log(1 - p0)
            log_h1 = n_violations * np.log(p_hat) + (n_obs - n_violations) * np.log(1 - p_hat)
            lr = -2 * (log_h0 - log_h1)

        pvalue = 1 - sp_stats.chi2.cdf(lr, df=1)
        return float(lr), float(pvalue)

    def christoffersen_test(
        self,
        violations: np.ndarray,
        confidence: float,
    ) -> tuple[float, float]:
        """
        Christoffersen (1998) conditional coverage test.

        Tests independence of violations (clustering of exceedances).
        Uses a Markov chain model of violation states.

        Parameters
        ----------
        violations : np.ndarray
            Boolean array (1=violation, 0=no violation).
        confidence : float
            Expected coverage level.

        Returns
        -------
        (statistic, pvalue)
        """
        v = np.asarray(violations, dtype=int)
        n = len(v)

        # Count transitions
        n00 = sum(1 for t in range(1, n) if v[t - 1] == 0 and v[t] == 0)
        n01 = sum(1 for t in range(1, n) if v[t - 1] == 0 and v[t] == 1)
        n10 = sum(1 for t in range(1, n) if v[t - 1] == 1 and v[t] == 0)
        n11 = sum(1 for t in range(1, n) if v[t - 1] == 1 and v[t] == 1)

        # Transition probabilities (full model)
        pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0

        # Unconditional probability (restricted model)
        pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0

        # Log-likelihoods
        eps = 1e-12
        log_full = (
            n00 * np.log(max(1 - pi01, eps))
            + n01 * np.log(max(pi01, eps))
            + n10 * np.log(max(1 - pi11, eps))
            + n11 * np.log(max(pi11, eps))
        )

        log_restricted = (
            (n00 + n10) * np.log(max(1 - pi, eps))
            + (n01 + n11) * np.log(max(pi, eps))
        )

        lr_ind = -2 * (log_restricted - log_full)
        pvalue = 1 - sp_stats.chi2.cdf(lr_ind, df=1)

        return float(lr_ind), float(pvalue)

    def full_backtest(
        self,
        returns: pd.Series,
        var_series: pd.Series,
        confidence: float = 0.99,
        method_name: str = "VaR backtest",
    ) -> BacktestResult:
        """
        Run full backtest: Kupiec + Christoffersen.

        Parameters
        ----------
        returns : pd.Series
            Actual spread changes.
        var_series : pd.Series
            VaR estimates (positive values = loss threshold).
        confidence : float
            Expected VaR confidence level.

        Convention: violation = |return| > VaR (exceedance in either direction).
        """
        # Align indices
        common_idx = returns.index.intersection(var_series.index)
        r = returns.loc[common_idx].values
        v = var_series.loc[common_idx].values

        n_obs = len(r)
        violations_arr = np.abs(r) > v
        n_violations = int(violations_arr.sum())
        expected = n_obs * (1 - confidence)
        actual_coverage = 1 - n_violations / n_obs if n_obs > 0 else 1.0

        # Kupiec test
        kupiec_stat, kupiec_p = self.kupiec_test(n_violations, n_obs, confidence)

        # Christoffersen test
        chris_stat, chris_p = self.christoffersen_test(
            violations_arr.astype(int), confidence
        )

        # Pass criteria: both p-values > 0.05
        passes = kupiec_p > 0.05 and chris_p > 0.05

        result = BacktestResult(
            method=method_name,
            n_observations=n_obs,
            n_violations=n_violations,
            expected_violations=expected,
            actual_coverage=actual_coverage,
            kupiec_stat=kupiec_stat,
            kupiec_pvalue=kupiec_p,
            christoffersen_stat=chris_stat,
            christoffersen_pvalue=chris_p,
            passes=passes,
        )

        logger.info(
            "Backtest: %d violations (expected %.1f), coverage=%.4f, passes=%s",
            n_violations, expected, actual_coverage, passes,
        )
        return result


class HistoricalBacktestAnalyzer(RiskAnalyzer):
    """
    RiskAnalyzer implementation that uses historical VaR + backtesting.
    """

    def __init__(self) -> None:
        self._bt = VaRBacktest()

    def fit(self, returns: pd.Series, confidence: float = 0.99) -> Self:
        from src.risk.var_engine import VaREngine
        result = VaREngine.historical_var(returns, confidence)
        self._result = RiskResult(
            method="historical",
            confidence=confidence,
            var=result["var"],
            es=result["es"],
        )
        return self

    def backtest(self, returns: pd.Series, var_series: pd.Series) -> BacktestResult:
        return self._bt.full_backtest(
            returns, var_series,
            confidence=self.result.confidence,
            method_name="historical_backtest",
        )
