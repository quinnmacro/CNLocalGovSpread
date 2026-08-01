"""
FIGARCH (Fractionally Integrated GARCH) with GPH long-memory estimator.

FIGARCH(p, d, q) extends GARCH by replacing the integer differencing operator (1-L)
with its fractional counterpart (1-L)^d, allowing for long-range dependence in
volatility — a stylised fact often observed in financial spread data.

The model:
    (1 - βL)·σ²_t = ω + [1 - βL - α(L)·(1-L)^d]·r²_t

where (1-L)^d = Σ_{k=0}^∞ π_k·L^k  with  π_0 = 1,  π_k = π_{k-1}·(k-1-d)/k.

Implementation notes:
- GPH (Geweke–Porter–Hudak) spectral estimator provides an initial d estimate
- Truncation of π-weights at `truncation_lag` to avoid O(T²) cost
- Parameters estimated via Gaussian QMLE with scipy L-BFGS-B
- AIC / BIC are valid (QMLE-based)
"""

from __future__ import annotations

from typing import Optional, Self

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import jarque_bera, linregress
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from src.core.base import NotFittedError, VolatilityModel
from src.core.config import get_settings
from src.core.logging_config import get_logger
from src.core.types import DiagnosticsResult, VolatilityResult

logger = get_logger(__name__)


class FIGARCHModel(VolatilityModel):
    """
    Fractionally Integrated GARCH with GPH long-memory estimator.

    Parameters
    ----------
    truncation_lag : Number of π-weight lags to retain (truncates infinite sum).
                     Higher = more accurate but slower. 500 is a good default.
    """

    def __init__(self, truncation_lag: int | None = None) -> None:
        self._truncation = truncation_lag or get_settings().model.figarch_truncation
        if self._truncation < 10:
            raise ValueError(f"truncation_lag must be >= 10, got {self._truncation}")

        self._returns: np.ndarray | None = None
        self._fitted_params: dict[str, float] | None = None
        self._loglikelihood: float | None = None

        logger.debug("FIGARCHModel created: truncation_lag=%d", self._truncation)

    # ------------------------------------------------------------------
    # VolatilityModel interface
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series) -> Self:
        """Fit FIGARCH via QMLE with GPH initialisation for d."""
        if len(returns) < 50:
            raise ValueError(f"FIGARCH requires at least 50 observations, got {len(returns)}")

        self._returns = returns.values.astype(np.float64)
        r = self._returns
        T = len(r)

        # Step 1: GPH estimator for initial d
        d_gph = self._gph_estimator(r)
        logger.info("GPH estimate of long-memory d = %.4f", d_gph)

        # Step 2: Initial parameter guesses
        r_var = np.var(r)
        omega_init = r_var * 0.01
        alpha_init = 0.1
        beta_init = 0.8
        mu_init = np.mean(r)

        x0 = np.array([mu_init, omega_init, alpha_init, beta_init, np.clip(d_gph, 0.05, 0.95)])

        # Bounds: μ (free), ω > 0, α ∈ [0, 0.99], β ∈ [0, 0.99], d ∈ [0, 1]
        bounds = [
            (None, None),          # μ
            (1e-8, None),          # ω
            (0.0, 0.99),           # α
            (0.0, 0.99),           # β
            (0.0, 1.0),            # d
        ]

        logger.info("Fitting FIGARCH via QMLE on %d observations...", T)

        result = optimize.minimize(
            self._neg_loglikelihood,
            x0,
            args=(r,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-8},
        )

        if not result.success:
            logger.warning("FIGARCH QMLE did not converge: %s", result.message)
        else:
            logger.info("FIGARCH QMLE converged in %d iterations", result.nit)

        mu_hat, omega_hat, alpha_hat, beta_hat, d_hat = result.x
        self._fitted_params = {
            "mu": float(mu_hat),
            "omega": float(omega_hat),
            "alpha": float(alpha_hat),
            "beta": float(beta_hat),
            "d": float(d_hat),
        }
        self._loglikelihood = -float(result.fun) * T

        # Compute conditional variance with fitted params
        cond_var = self._compute_conditional_variance(r, self._fitted_params)
        cond_vol = np.sqrt(np.maximum(cond_var, 1e-12))

        # Standardised residuals
        demeaned = r - mu_hat
        std_resid = demeaned / cond_vol

        # Align with pandas index
        idx = returns.index
        cond_vol_series = pd.Series(cond_vol, index=idx, name="cond_vol")
        std_resid_series = pd.Series(std_resid, index=idx, name="std_resid")

        # AIC / BIC (QMLE-based)
        k = len(self._fitted_params)
        aic = 2 * k - 2 * self._loglikelihood
        bic = k * np.log(T) - 2 * self._loglikelihood

        self._result = VolatilityResult(
            model_name=f"FIGARCH(trunc={self._truncation})",
            params=self._fitted_params,
            conditional_volatility=cond_vol_series,
            standardized_residuals=std_resid_series,
            aic=float(aic),
            bic=float(bic),
            loglikelihood=float(self._loglikelihood),
            converged=result.success,
        )

        logger.info(
            "FIGARCH fit: d=%.4f, α=%.4f, β=%.4f, AIC=%.2f, BIC=%.2f",
            d_hat, alpha_hat, beta_hat, aic, bic,
        )
        return self

    def conditional_variance(self) -> pd.Series:
        """Return in-sample conditional variance σ²."""
        if not self.is_fitted:
            raise NotFittedError("Call fit() before accessing conditional variance.")
        return self._result.conditional_volatility ** 2

    def forecast(self, steps: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        Forecast conditional volatility for `steps` periods ahead.

        Uses mean-reversion to unconditional variance:
        σ²_{T+h} = ω/(1-β) + β^h * (σ²_T - ω/(1-β))
        """
        if not self.is_fitted:
            raise NotFittedError("Call fit() before forecasting.")

        params = self._fitted_params
        assert params is not None

        omega = params["omega"]
        beta = params["beta"]
        mu = params["mu"]

        cond_vol = self._result.conditional_volatility.values
        sigma2_T = cond_vol[-1] ** 2

        sigma2_unc = omega / (1.0 - beta) if beta < 1 else sigma2_T

        forecast_var = np.zeros(steps)
        for h in range(steps):
            forecast_var[h] = sigma2_unc + (beta ** (h + 1)) * (sigma2_T - sigma2_unc)

        forecast_vol = np.sqrt(np.maximum(forecast_var, 0.0))
        forecast_mean = np.full(steps, mu)

        return forecast_mean, forecast_vol

    def diagnose(self) -> DiagnosticsResult:
        """Run diagnostic tests on standardised residuals."""
        if not self.is_fitted:
            raise NotFittedError("Call fit() before running diagnostics.")

        std_resid = self._result.standardized_residuals.dropna().values.astype(float)
        n = len(std_resid)

        # Ljung-Box Q-test
        lb_df = acorr_ljungbox(std_resid, lags=[min(10, max(1, n // 5))])
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
    # GPH spectral estimator
    # ------------------------------------------------------------------

    @staticmethod
    def _gph_estimator(x: np.ndarray, bandwidth: int | None = None) -> float:
        """
        Geweke–Porter–Hudak (GPH) spectral estimator for long-memory parameter d.

        Regresses log(periodogram) on log(4·sin²(ω_j / 2)) for j = 1, ..., m
        where m = floor(T^0.5).  The OLS slope is −d.
        """
        T = len(x)
        if bandwidth is None:
            bandwidth = int(np.floor(np.sqrt(T)))
        m = max(min(bandwidth, T // 2 - 1), 5)

        # Use FFT for efficient periodogram computation
        x_dm = x - np.mean(x)
        fft_vals = np.fft.fft(x_dm)
        periodogram = (np.abs(fft_vals[:m + 1]) ** 2) / (2.0 * np.pi * T)
        periodogram = periodogram[1:m + 1]  # drop frequency 0

        freqs = 2.0 * np.pi * np.arange(1, m + 1) / T
        log_freq = np.log(4.0 * np.sin(freqs / 2.0) ** 2)
        log_per = np.log(np.maximum(periodogram, 1e-30))

        slope, _, _, _, _ = linregress(log_freq, log_per)
        d_est = -float(slope)
        return float(np.clip(d_est, 0.0, 1.0))

    # ------------------------------------------------------------------
    # FIGARCH conditional variance
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_pi_weights(d: float, max_lag: int) -> np.ndarray:
        """
        Compute the fractional difference filter π-weights.

        (1 − L)^d = Σ_{k=0}^∞ π_k L^k
        π_0 = 1,  π_k = π_{k−1} · (k − 1 − d) / k   for k ≥ 1
        """
        pi = np.zeros(max_lag + 1)
        pi[0] = 1.0
        for k in range(1, max_lag + 1):
            pi[k] = pi[k - 1] * (k - 1 - d) / k
        return pi

    def _compute_conditional_variance(
        self, returns: np.ndarray, params: dict[str, float]
    ) -> np.ndarray:
        """
        Compute FIGARCH conditional variance via truncated filter recursion.

        σ²_t = ω + β·σ²_{t−1} + α·[r²_{t−1} − Σ_{k=1}^{min(t,trunc)} π_k·r²_{t−1−k}]
        """
        T = len(returns)
        mu = params["mu"]
        omega = params["omega"]
        alpha = params["alpha"]
        beta = params["beta"]
        d = params["d"]

        eps = returns - mu
        eps2 = eps ** 2

        trunc = min(self._truncation, T - 1)
        pi_weights = self._compute_pi_weights(d, trunc)

        sigma2 = np.zeros(T)
        sigma2_unc = omega / (1.0 - beta) if abs(beta) < 1.0 else np.var(eps)
        sigma2[0] = sigma2_unc

        for t in range(1, T):
            max_k = min(t, trunc + 1)
            frac_diff_sum = 0.0
            for k in range(max_k):
                frac_diff_sum += pi_weights[k] * eps2[t - 1 - k]

            sigma2[t] = omega + beta * sigma2[t - 1] + alpha * (eps2[t - 1] - frac_diff_sum)
            if sigma2[t] < 1e-12:
                sigma2[t] = 1e-12

        return sigma2

    # ------------------------------------------------------------------
    # QMLE objective
    # ------------------------------------------------------------------

    def _neg_loglikelihood(self, x: np.ndarray, returns: np.ndarray) -> float:
        """Negative Gaussian quasi-log-likelihood (per observation)."""
        mu, omega, alpha, beta, d = x

        if alpha + beta >= 1.0 or omega <= 0:
            return 1e10

        params = {"mu": mu, "omega": omega, "alpha": alpha, "beta": beta, "d": d}

        try:
            sigma2 = self._compute_conditional_variance(returns, params)
        except (ValueError, FloatingPointError):
            return 1e10

        eps = returns - mu
        sigma2 = np.maximum(sigma2, 1e-12)
        nll = 0.5 * np.mean(np.log(sigma2) + eps ** 2 / sigma2)

        if np.isnan(nll) or np.isinf(nll):
            return 1e10

        return float(nll)
