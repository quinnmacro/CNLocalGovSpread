"""
RiskMetrics EWMA (Exponentially Weighted Moving Average) volatility model
with automatic λ calibration via QLIKE loss.

EWMA is the simplest volatility model — a special case of IGARCH(1,1)
where α + β = 1. It has one free parameter (λ, the decay factor) and
no mean reversion, making it useful as a benchmark.

EWMA recursion:
    σ²[t] = λ · σ²[t-1] + (1−λ) · r²[t-1]

QLIKE calibration (Patton 2011):
    L(λ) = (1/T) Σ [ r²[t]/σ²[t] − log(r²[t]/σ²[t]) − 1 ]

QLIKE is a robust loss for variance evaluation because it is invariant
to scale and well-behaved even when realised variance proxies are noisy.

Note: AIC / BIC are NOT defined for EWMA (not MLE) — set to None.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from src.core.base import NotFittedError, VolatilityModel
from src.core.logging_config import get_logger
from src.core.types import DiagnosticsResult, VolatilityResult

logger = get_logger(__name__)


class EWMAModel(VolatilityModel):
    """
    RiskMetrics EWMA with automatic λ calibration via QLIKE loss.

    Parameters
    ----------
    lambda_param :
        Decay factor in [0.80, 0.99].
        If None (default), auto-calibrate via QLIKE grid search.
        Common choices: 0.94 (RiskMetrics default), 0.97 (longer memory).
    """

    def __init__(self, lambda_param: float | None = None) -> None:
        if lambda_param is not None:
            if not (0.5 < lambda_param < 1.0):
                raise ValueError(f"lambda_param must be in (0.5, 1.0), got {lambda_param}")
        self.lambda_param = lambda_param
        self._calibrated_lambda: float | None = None
        self._returns: np.ndarray | None = None

        logger.debug("EWMAModel created: lambda=%s", lambda_param or "auto")

    # ------------------------------------------------------------------
    # VolatilityModel interface
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series) -> Self:
        """Fit EWMA to a return series, optionally calibrating λ."""
        if len(returns) < 10:
            raise ValueError(f"EWMA requires at least 10 observations, got {len(returns)}")

        self._returns = returns.values.astype(np.float64)
        r = self._returns

        if self.lambda_param is None:
            lam = self._calibrate_lambda(r)
            self._calibrated_lambda = lam
            logger.info("EWMA auto-calibrated λ = %.6f via QLIKE", lam)
        else:
            lam = self.lambda_param
            logger.info("EWMA using fixed λ = %.4f", lam)

        cond_var = self._ewma_variance(r, lam)
        cond_vol = np.sqrt(np.maximum(cond_var, 1e-12))

        std_resid = r / cond_vol

        idx = returns.index
        cond_vol_series = pd.Series(cond_vol, index=idx, name="cond_vol")
        std_resid_series = pd.Series(std_resid, index=idx, name="std_resid")

        self._result = VolatilityResult(
            model_name=f"EWMA(λ={lam:.4f})",
            params={"lambda": lam},
            conditional_volatility=cond_vol_series,
            standardized_residuals=std_resid_series,
            aic=None,
            bic=None,
            loglikelihood=None,
            converged=True,
        )

        logger.info("EWMA fit complete: λ=%.4f, %d observations", lam, len(r))
        return self

    def conditional_variance(self) -> pd.Series:
        """Return in-sample conditional variance σ²."""
        if not self.is_fitted:
            raise NotFittedError("Call fit() before accessing conditional variance.")
        return self._result.conditional_volatility ** 2

    def forecast(self, steps: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        EWMA forecast: σ² is constant (random walk in variance).

        σ²_{T+h} = σ²_T for all h ≥ 1.
        Mean forecast = 0 (EWMA has no mean equation).
        """
        if not self.is_fitted:
            raise NotFittedError("Call fit() before forecasting.")

        last_vol = self._result.conditional_volatility.iloc[-1]
        forecast_vol = np.full(steps, last_vol)
        forecast_mean = np.zeros(steps)

        return forecast_mean, forecast_vol

    def diagnose(self) -> DiagnosticsResult:
        """Run diagnostic tests on standardised residuals."""
        if not self.is_fitted:
            raise NotFittedError("Call fit() before running diagnostics.")

        std_resid = self._result.standardized_residuals.dropna().values.astype(float)
        n = len(std_resid)

        # Ljung-Box Q-test
        max_lags = min(10, max(1, n // 5))
        lb_df = acorr_ljungbox(std_resid, lags=[max_lags])
        lb_stat = float(lb_df["lb_stat"].iloc[0])
        lb_pvalue = float(lb_df["lb_pvalue"].iloc[0])

        # ARCH-LM test
        nlags_arch = min(5, max(1, n // 10))
        arch_lm = het_arch(std_resid, nlags=nlags_arch)
        arch_stat = float(arch_lm[2])
        arch_pvalue = float(arch_lm[3])

        # Jarque-Bera
        jb_res = jarque_bera(std_resid)
        jb_stat = float(jb_res.statistic)
        jb_pvalue = float(jb_res.pvalue)

        return DiagnosticsResult(
            ljung_box_stat=lb_stat,
            ljung_box_pvalue=lb_pvalue,
            arch_lm_stat=arch_stat,
            arch_lm_pvalue=arch_pvalue,
            jarque_bera_stat=jb_stat,
            jarque_bera_pvalue=jb_pvalue,
            n_obs=n,
        )

    # ------------------------------------------------------------------
    # QLIKE calibration
    # ------------------------------------------------------------------

    def _calibrate_lambda(
        self,
        returns: np.ndarray,
        grid: tuple[float, float] = (0.80, 0.99),
    ) -> float:
        """
        Calibrate λ by minimising QLIKE loss over the return series.

        QLIKE (Patton 2011):
            L(λ) = (1/T) Σ [ r²_t / σ²_t(λ) − log(r²_t / σ²_t(λ)) − 1 ]
        """
        lam_low, lam_high = grid

        def qlike_loss(lam: float) -> float:
            sigma2 = self._ewma_variance(returns, lam)
            r2 = returns ** 2
            mask = (sigma2 > 1e-12) & (r2 > 1e-12)
            if mask.sum() < 10:
                return 1e10
            ratio = r2[mask] / sigma2[mask]
            loss = np.mean(ratio - np.log(ratio) - 1.0)
            return float(loss)

        result = optimize.minimize_scalar(
            qlike_loss,
            bounds=(lam_low, lam_high),
            method="bounded",
            options={"xatol": 1e-8, "maxiter": 500},
        )

        if not result.success:
            logger.warning("QLIKE calibration failed, falling back to RiskMetrics λ=0.94")
            return 0.94

        return float(result.x)

    # ------------------------------------------------------------------
    # EWMA recursion
    # ------------------------------------------------------------------

    @staticmethod
    def _ewma_variance(returns: np.ndarray, lam: float) -> np.ndarray:
        """
        Compute EWMA conditional variance series.

        σ²[0] = var(r[:5])   — short-sample initialisation
        σ²[t] = λ · σ²[t-1] + (1−λ) · r²[t-1]   for t ≥ 1
        """
        T = len(returns)
        sigma2 = np.zeros(T)

        init_len = min(5, T)
        sigma2[0] = np.var(returns[:init_len]) if init_len > 1 else returns[0] ** 2

        one_minus_lam = 1.0 - lam
        for t in range(1, T):
            sigma2[t] = lam * sigma2[t - 1] + one_minus_lam * returns[t - 1] ** 2

        sigma2 = np.maximum(sigma2, 1e-12)
        return sigma2
