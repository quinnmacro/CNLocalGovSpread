"""
Heterogeneous Autoregressive Realized Volatility (HAR-RV) model.

Corsi (2009) — A simple and powerful forecast model for realized volatility
that captures the heterogeneous nature of market participants operating on
different time horizons.

HAR-RV recursion:
    RV_t = β_0 + β_d·RV_{t-1} + β_w·RV^{(w)}_{t-1} + β_m·RV^{(m)}_{t-1} + ε_t

where:
    RV_t        = realised variance at day t (proxy: Σ r² over rolling window)
    RV^{(w)}_t  = (1/5) Σ_{i=0}^{4} RV_{t-i}   (weekly average)
    RV^{(m)}_t  = (1/22) Σ_{i=0}^{21} RV_{t-i}  (monthly average)

Key insight: unlike GARCH's single exponential decay, the HAR structure
produces long-memory-like behaviour through the superposition of three
distinct time scales — daily traders, weekly traders, and monthly investors.

Note: AIC/BIC are approximate (OLS-based, not MLE), but comparable for
model selection purposes.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from src.core.base import NotFittedError, VolatilityModel
from src.core.logging_config import get_logger
from src.core.types import DiagnosticsResult, VolatilityResult

logger = get_logger(__name__)


class HARRVModel(VolatilityModel):
    """
    Heterogeneous Autoregressive Realized Volatility model (Corsi 2009).

    Parameters
    ----------
    daily_window : int
        Window for daily RV proxy (default 1, i.e., r²_t).
    weekly_window : int
        Window for weekly average (default 5).
    monthly_window : int
        Window for monthly average (default 22).
    """

    def __init__(
        self,
        daily_window: int = 1,
        weekly_window: int = 5,
        monthly_window: int = 22,
    ) -> None:
        self.daily_window = daily_window
        self.weekly_window = weekly_window
        self.monthly_window = monthly_window
        self._returns: np.ndarray | None = None

        logger.debug(
            "HARRVModel created: d=%d, w=%d, m=%d",
            daily_window, weekly_window, monthly_window,
        )

    # ------------------------------------------------------------------
    # VolatilityModel interface
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series) -> Self:
        """Fit HAR-RV to a return series via OLS."""
        if len(returns) < self.monthly_window + 5:
            raise ValueError(
                f"HAR-RV requires at least {self.monthly_window + 5} observations, "
                f"got {len(returns)}"
            )

        self._returns = returns.values.astype(np.float64)
        r = self._returns
        T = len(r)

        # Step 1: Compute realised variance proxy
        # Using rolling sum of squared returns as daily RV proxy
        rv = pd.Series(r ** 2, index=returns.index)

        # Rolling averages for weekly and monthly components
        rv_daily = rv.copy()
        rv_weekly = rv.rolling(window=self.weekly_window, min_periods=1).mean()
        rv_monthly = rv.rolling(window=self.monthly_window, min_periods=1).mean()

        # Step 2: Build design matrix (skip first monthly_window observations)
        start = self.monthly_window
        n = T - start

        y = rv_daily.values[start:]
        X = np.column_stack([
            np.ones(n),                        # intercept
            rv_daily.values[start - 1:T - 1],  # daily lag
            rv_weekly.values[start - 1:T - 1], # weekly lag
            rv_monthly.values[start - 1:T - 1], # monthly lag
        ])

        # Step 3: OLS estimation
        beta, residuals_sum, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        fitted = X @ beta
        resid = y - fitted

        # Step 4: Compute conditional volatility
        # For the full series, use fitted values where available,
        # and rolling RV proxy for the initial period
        cond_var_full = np.full(T, np.nan)
        cond_var_full[start:] = np.maximum(fitted, 1e-12)

        # Fill the initial period with rolling variance
        rolling_var = pd.Series(r ** 2).rolling(
            window=self.monthly_window, min_periods=1
        ).mean().values
        mask = np.isnan(cond_var_full)
        cond_var_full[mask] = rolling_var[mask]
        cond_var_full = np.maximum(cond_var_full, 1e-12)

        cond_vol = np.sqrt(cond_var_full)
        std_resid = r / cond_vol

        # Step 5: Compute model statistics
        r_squared = float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2))
        sigma2 = np.sum(resid ** 2) / n
        loglik = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) + 1)
        k = 4  # number of parameters
        aic = 2 * k - 2 * loglik
        bic = k * np.log(n) - 2 * loglik

        idx = returns.index
        cond_vol_series = pd.Series(cond_vol, index=idx, name="cond_vol")
        std_resid_series = pd.Series(std_resid, index=idx, name="std_resid")

        params = {
            "beta_0": float(beta[0]),
            "beta_d": float(beta[1]),
            "beta_w": float(beta[2]),
            "beta_m": float(beta[3]),
            "r_squared": r_squared,
        }

        self._result = VolatilityResult(
            model_name=f"HAR-RV({self.daily_window},{self.weekly_window},{self.monthly_window})",
            params=params,
            conditional_volatility=cond_vol_series,
            standardized_residuals=std_resid_series,
            aic=float(aic),
            bic=float(bic),
            loglikelihood=float(loglik),
            converged=True,
        )

        # Store intermediate results for API
        self._rv_daily = pd.Series(rv_daily.values, index=idx)
        self._rv_weekly = pd.Series(rv_weekly.values, index=idx)
        self._rv_monthly = pd.Series(rv_monthly.values, index=idx)
        self._r_squared = r_squared

        logger.info(
            "HAR-RV fit complete: R²=%.4f, AIC=%.2f, β=[%.4f, %.4f, %.4f, %.4f]",
            r_squared, aic, beta[0], beta[1], beta[2], beta[3],
        )
        return self

    def conditional_variance(self) -> pd.Series:
        """Return in-sample conditional variance σ²."""
        if not self.is_fitted:
            raise NotFittedError("Call fit() before accessing conditional variance.")
        return self._result.conditional_volatility ** 2

    def forecast(self, steps: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        HAR-RV iterative forecast.

        Iteratively apply the HAR equation to forecast future RV.
        """
        if not self.is_fitted:
            raise NotFittedError("Call fit() before forecasting.")

        r = self._returns
        assert r is not None

        beta = self._result.params
        b0 = beta["beta_0"]
        bd = beta["beta_d"]
        bw = beta["beta_w"]
        bm = beta["beta_m"]

        # Last observed values
        rv_vals = r[-self.monthly_window:] ** 2
        rv_d = float(rv_vals[-1])
        rv_w = float(np.mean(rv_vals[-self.weekly_window:]))
        rv_m = float(np.mean(rv_vals))

        forecast_vol = np.zeros(steps)
        forecast_mean = np.zeros(steps)

        history = list(rv_vals)  # full history for rolling computation

        for h in range(steps):
            rv_forecast = b0 + bd * rv_d + bw * rv_w + bm * rv_m
            rv_forecast = max(rv_forecast, 1e-12)
            forecast_vol[h] = np.sqrt(rv_forecast)

            # Update history for next step
            history.append(rv_forecast)
            rv_d = rv_forecast
            rv_w = float(np.mean(history[-self.weekly_window:]))
            rv_m = float(np.mean(history[-self.monthly_window:]))

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
