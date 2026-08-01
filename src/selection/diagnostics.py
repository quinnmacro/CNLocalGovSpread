"""
Statistical diagnostics for model residuals.

Tests: Ljung-Box (autocorrelation), ARCH-LM (heteroskedasticity),
       Jarque-Bera (normality).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from src.core.logging_config import get_logger
from src.core.types import DiagnosticsResult

logger = get_logger(__name__)


def ljung_box_test(residuals: pd.Series | np.ndarray, lags: int = 10) -> tuple[float, float]:
    """
    Ljung-Box test for autocorrelation.

    H0: no autocorrelation up to lag `lags`.
    Low p-value => significant autocorrelation in residuals.
    """
    r = np.asarray(residuals, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < lags + 5:
        lags = max(1, len(r) // 5 - 1)

    result = acorr_ljungbox(r, lags=[lags], return_df=True)
    stat = float(result["lb_stat"].iloc[0])
    pvalue = float(result["lb_pvalue"].iloc[0])
    return stat, pvalue


def arch_lm_test(residuals: pd.Series | np.ndarray, lags: int = 10) -> tuple[float, float]:
    """
    ARCH-LM test for remaining heteroskedasticity.

    H0: no ARCH effects (squared residuals are not autocorrelated).
    Low p-value => model has not fully captured volatility clustering.
    """
    r = np.asarray(residuals, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < lags + 5:
        lags = max(1, len(r) // 5 - 1)

    result = het_arch(r, nlags=lags)
    # result = (lm_stat, lm_pvalue, f_stat, f_pvalue)
    stat = float(result[0])
    pvalue = float(result[1])
    return stat, pvalue


def jarque_bera_test(residuals: pd.Series | np.ndarray) -> tuple[float, float]:
    """
    Jarque-Bera test for normality.

    H0: residuals are normally distributed.
    Low p-value => non-normal (fat tails, skewness).
    """
    r = np.asarray(residuals, dtype=float)
    r = r[~np.isnan(r)]

    stat, pvalue = sp_stats.jarque_bera(r)
    return float(stat), float(pvalue)


def compute_diagnostics(
    residuals: pd.Series | np.ndarray,
    lags: int = 10,
) -> DiagnosticsResult:
    """
    Run all three diagnostic tests and return a DiagnosticsResult.

    Parameters
    ----------
    residuals : array-like
        Standardized residuals from a fitted model.
    lags : int
        Number of lags for Ljung-Box and ARCH-LM tests.
    """
    r = np.asarray(residuals, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)

    lb_stat, lb_p = ljung_box_test(r, lags)
    arch_stat, arch_p = arch_lm_test(r, lags)
    jb_stat, jb_p = jarque_bera_test(r)

    diag = DiagnosticsResult(
        ljung_box_stat=lb_stat,
        ljung_box_pvalue=lb_p,
        arch_lm_stat=arch_stat,
        arch_lm_pvalue=arch_p,
        jarque_bera_stat=jb_stat,
        jarque_bera_pvalue=jb_p,
        n_obs=n,
    )

    logger.info(
        "Diagnostics (n=%d): LB p=%.4f, ARCH p=%.4f, JB p=%.4f",
        n, lb_p, arch_p, jb_p,
    )
    return diag
