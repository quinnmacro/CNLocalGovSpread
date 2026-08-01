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
- **Vectorised convolution** via FFT: O(T log T) instead of O(T·K) per likelihood eval
- AIC / BIC are valid (QMLE-based)

Performance:
- v4.0 had O(T·K) pure-Python double loop: ~30s for T=500, K=500
- v4.1 vectorises the convolution to O(T log T) + O(T) recursion: ~2-3s
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.signal import fftconvolve
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
            (None, None),   # μ
            (1e-8, None),   # ω
            (0.0, 0.99),    # α
            (0.0, 0.99),    # β
            (0.0, 1.0),     # d
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
            loglikelihood=self._loglikelihood,
            converged=result.success,
        )

        logger.info(
            "FIGARCH fit complete: d=%.4f, AIC=%.2f, BIC=%.2f",
            d_hat, aic, bic,
        )
        return self

    def conditional_variance(self) -> pd.Series:
        """Return the in-sample conditional variance σ²."""
        if not self.is_fitted:
            raise NotFittedError("Call fit() before accessing conditional variance.")
        return self._result.conditional_volatility ** 2

    def forecast(self, steps: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        Forecast conditional mean and volatility.

        For FIGARCH, the long-memory filter makes analytical forecasts complex,
        so we use the conditional variance's persistence structure.

        Returns
        -------
        forecast_mean : ndarray of shape (steps,)
        forecast_vol  : ndarray of shape (steps,) — σ (not σ²)
        """
        if not self.is_fitted:
            raise NotFittedError("Call fit() before forecasting.")

        params = self._fitted_params
        assert params is not None
        mu = params["mu"]
        omega = params["omega"]
        beta = params["beta"]
        alpha = params["alpha"]

        last_var = float(self._result.conditional_volatility.iloc[-1] ** 2)
        unc_var = omega / max(1.0 - beta - alpha, 1e-6)  # unconditional variance proxy

        forecast_mean = np.full(steps, mu)
        forecast_var = np.zeros(steps)

        # Simple variance recursion: σ²_{t+h} ≈ ω + (α+β)·σ²_{t+h-1}
        # (Approximate — ignores the fractional difference term in forecast)
        persistence = alpha + beta
        for h in range(steps):
            if h == 0:
                forecast_var[h] = omega + persistence * last_var
            else:
                forecast_var[h] = omega + persistence * forecast_var[h - 1]

        return forecast_mean, np.sqrt(np.maximum(forecast_var, 1e-12))

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
    # FIGARCH conditional variance — VECTORIZED
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
        # Vectorised via cumulative product of (k-1-d)/k ratio
        ks = np.arange(1, max_lag + 1, dtype=np.float64)
        ratios = (ks - 1.0 - d) / ks
        pi[1:] = np.cumprod(ratios)
        return pi

    def _compute_conditional_variance(
        self, returns: np.ndarray, params: dict[str, float]
    ) -> np.ndarray:
        """
        Compute FIGARCH conditional variance via FFT-accelerated convolution.

        σ²_t = ω + β·σ²_{t−1} + α·[ε²_{t−1} − z_{t−1}]

        where z_t = Σ_{k=0}^{K} π_k · ε²_{t−k}  (truncated convolution).

        Performance: O(T log T) for FFT convolution + O(T) recursion,
        vs. O(T·K) pure-Python double loop in v4.0.
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

        # ── Vectorised convolution via FFT ──
        # z_full[t] = Σ_k pi[k] · eps²[t-k]  for all t
        # fftconvolve(eps², pi, mode='full') gives length T+trunc+1
        conv_full = fftconvolve(eps2, pi_weights, mode="full")
        # conv_full[m] = Σ_k pi[k] · eps²[m-k] for valid indices
        # We need z[t] = Σ_k pi[k] · eps²[t-k] for t = 0, ..., T-1
        # This maps to conv_full[0:T] (since pi starts at index 0)
        z_all = conv_full[:T]

        # ── Define recursion input: c[t] = ω + α·(ε²_t − z_t) ──
        c = omega + alpha * (eps2 - z_all)

        # ── Unrolled recursion ──
        # σ²_t = c[t-1] + β · σ²_{t-1}   for t = 1, ..., T-1
        # σ²_0 = σ²_unc = ω / (1-β)
        sigma2 = np.empty(T)
        sigma2_unc = omega / max(1.0 - beta, 1e-12)

        # Unroll: σ²_t = Σ_{j=0}^{t-1} β^j · c[t-1-j] + β^t · σ²_0
        # Efficient computation via running cumulative sum:
        #   s_0 = σ²_0,  s_t = β · s_{t-1} + c[t-1]
        # Then σ²_t = s_t
        s = sigma2_unc
        sigma2[0] = s
        for t in range(1, T):
            s = beta * s + c[t - 1]
            sigma2[t] = max(s, 1e-12)

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
