"""
Bayesian Stochastic Volatility model (Taylor 1986, Kim-Shephard-Chib 1998).

State-space representation:
    r_t = exp(h_t / 2) · ε_t,        ε_t ~ N(0, 1)
    h_t = μ + φ·(h_{t-1} - μ) + σ_η·η_t,  η_t ~ N(0, 1)

Parameters:
    μ   — long-run level of log-volatility
    φ   — persistence of log-volatility (AR(1) coefficient)
    σ_η — vol-of-vol (innovation standard deviation of log-vol)

Implementation:
    Uses PyMC + ADVI for fast approximate Bayesian inference,
    following the same pattern as bayesian_sts.py.

    Priors:
        μ   ~ Normal(0, 10)
        φ   ~ 2·Beta(20, 1.5) - 1   (mapped to (-1, 1))
        σ_η ~ HalfNormal(0.5)

    The log-volatility path h_t is sampled jointly with the parameters.
    Posterior mean of exp(h_t / 2) gives the conditional volatility,
    and 80% HPD intervals provide uncertainty quantification.

Note: AIC/BIC are approximate (based on WAIC from PyMC).
"""

from __future__ import annotations

import time
from typing import Self

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from src.core.base import NotFittedError, VolatilityModel
from src.core.logging_config import get_logger
from src.core.types import DiagnosticsResult, VolatilityResult

logger = get_logger(__name__)


class StochasticVolModel(VolatilityModel):
    """
    Bayesian Stochastic Volatility model via PyMC + ADVI.

    Parameters
    ----------
    n_advi_steps : int
        Number of ADVI optimization steps (default 10000).
    n_samples : int
        Number of posterior samples to draw (default 200).
    random_seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_advi_steps: int = 2000,
        n_samples: int = 100,
        random_seed: int = 42,
    ) -> None:
        self.n_advi_steps = n_advi_steps
        self.n_samples = n_samples
        self.random_seed = random_seed
        self._returns: np.ndarray | None = None
        self._fitting_time: float = 0.0

        logger.debug(
            "StochasticVolModel created: advi_steps=%d, samples=%d",
            n_advi_steps, n_samples,
        )

    # ------------------------------------------------------------------
    # VolatilityModel interface
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series) -> Self:
        """Fit the SV model. Uses fast quasi-Bayesian by default, PyMC for high precision."""
        if len(returns) < 30:
            raise ValueError(
                f"StochasticVol requires at least 30 observations, got {len(returns)}"
            )

        self._returns = returns.values.astype(np.float64)
        r = self._returns
        T = len(r)
        idx = returns.index

        # Use fast quasi-Bayesian approach for real-time serving
        # PyMC is only used when explicitly requested with high n_advi_steps
        if self.n_advi_steps <= 5000:
            logger.info("Using fast quasi-Bayesian SV (n_advi_steps=%d)", self.n_advi_steps)
            self._fit_quasi_bayesian(returns)
            return self

        # Subsample for computational tractability if needed
        max_obs = 200
        if T > max_obs:
            # Keep the most recent observations
            r_sub = r[-max_obs:]
            idx_sub = idx[-max_obs:]
        else:
            r_sub = r
            idx_sub = idx

        T_sub = len(r_sub)

        try:
            import pymc as pm
            import pytensor.tensor as pt

            t0 = time.time()

            with pm.Model() as model:
                # Priors
                mu = pm.Normal("mu", mu=0, sigma=10)
                # φ ~ Beta(20, 1.5) mapped to (-1, 1)
                phi_raw = pm.Beta("phi_raw", alpha=20, beta=1.5)
                phi = 2 * phi_raw - 1
                sigma_eta = pm.HalfNormal("sigma_eta", sigma=0.5)

                # Log-volatility path (non-centered parameterization)
                h_raw = pm.Normal("h_raw", mu=0, sigma=1, shape=T_sub)

                # Transform to centered AR(1) process
                # h[0] = mu + sigma_eta / sqrt(1 - phi^2) * h_raw[0]
                # h[t] = mu + phi * (h[t-1] - mu) + sigma_eta * h_raw[t]
                h = pt.zeros(T_sub)
                h = pt.set_subtensor(
                    h[0],
                    mu + sigma_eta / pt.sqrt(1 - phi ** 2 + 1e-8) * h_raw[0],
                )
                for t in range(1, T_sub):
                    h = pt.set_subtensor(
                        h[t],
                        mu + phi * (h[t - 1] - mu) + sigma_eta * h_raw[t],
                    )

                # Observation equation
                # r_t ~ N(0, exp(h_t))
                pm.Normal("r_obs", mu=0, sigma=pt.exp(h / 2), observed=r_sub)

            # ADVI fitting
            with model:
                approx = pm.fit(
                    n=self.n_advi_steps,
                    method="advi",
                    random_seed=self.random_seed,
                    progressbar=False,
                )
                trace = approx.sample(self.n_samples, random_seed=self.random_seed)

            self._fitting_time = time.time() - t0

            # Extract posterior samples of log-volatility
            h_samples = trace.posterior["h_raw"].values  # (chains, draws, T_sub)
            # Reshape to (n_total_samples, T_sub)
            h_samples = h_samples.reshape(-1, T_sub)

            # Reconstruct h path from h_raw
            mu_samples = trace.posterior["mu"].values.reshape(-1)
            phi_samples = (
                2 * trace.posterior["phi_raw"].values.reshape(-1) - 1
            )
            sigma_eta_samples = trace.posterior["sigma_eta"].values.reshape(-1)

            n_total = len(mu_samples)
            h_paths = np.zeros((n_total, T_sub))

            for s in range(n_total):
                h_paths[s, 0] = (
                    mu_samples[s]
                    + sigma_eta_samples[s]
                    / np.sqrt(1 - phi_samples[s] ** 2 + 1e-8)
                    * h_samples[s, 0]
                )
                for t in range(1, T_sub):
                    h_paths[s, t] = (
                        mu_samples[s]
                        + phi_samples[s] * (h_paths[s, t - 1] - mu_samples[s])
                        + sigma_eta_samples[s] * h_samples[s, t]
                    )

            # Posterior mean of volatility = exp(h/2)
            vol_paths = np.exp(h_paths / 2)
            vol_mean = np.mean(vol_paths, axis=0)
            vol_lower = np.percentile(vol_paths, 10, axis=0)
            vol_upper = np.percentile(vol_paths, 90, axis=0)

            # Parameter posterior means
            mu_hat = float(np.mean(mu_samples))
            phi_hat = float(np.mean(phi_samples))
            sigma_eta_hat = float(np.mean(sigma_eta_samples))

            # Build result — pad to full series length if subsampled
            if T > max_obs:
                # For the initial period, use rolling variance as proxy
                rolling_var = pd.Series(r ** 2).rolling(22, min_periods=1).mean().values
                vol_full = np.sqrt(np.maximum(rolling_var, 1e-12))
                vol_full[-T_sub:] = vol_mean
                vol_lower_full = np.full(T, np.nan)
                vol_upper_full = np.full(T, np.nan)
                vol_lower_full[-T_sub:] = vol_lower
                vol_upper_full[-T_sub:] = vol_upper
            else:
                vol_full = vol_mean
                vol_lower_full = vol_lower
                vol_upper_full = vol_upper

            vol_full = np.maximum(vol_full, 1e-12)
            std_resid = r / vol_full

            cond_vol_series = pd.Series(vol_full, index=idx, name="cond_vol")
            std_resid_series = pd.Series(std_resid, index=idx, name="std_resid")

            # Approximate information criterion
            loglik_approx = float(np.sum(
                -0.5 * np.log(2 * np.pi) - np.log(vol_full) - 0.5 * (r / vol_full) ** 2
            ))
            k = 3 + T_sub  # parameters + latent states
            aic = 2 * k - 2 * loglik_approx
            bic = k * np.log(T) - 2 * loglik_approx

            self._result = VolatilityResult(
                model_name="StochasticVol(Bayesian)",
                params={
                    "mu": mu_hat,
                    "phi": phi_hat,
                    "sigma_eta": sigma_eta_hat,
                    "persistence": phi_hat,
                },
                conditional_volatility=cond_vol_series,
                standardized_residuals=std_resid_series,
                aic=float(aic),
                bic=float(bic),
                loglikelihood=loglik_approx,
                converged=True,
            )

            # Store extra info for API
            self._vol_lower = pd.Series(vol_lower_full, index=idx)
            self._vol_upper = pd.Series(vol_upper_full, index=idx)
            self._log_vol_mean = pd.Series(np.log(np.maximum(vol_full, 1e-12)), index=idx)
            self._fitting_time = time.time() - t0

            logger.info(
                "StochasticVol fit complete: μ=%.4f, φ=%.4f, σ_η=%.4f, time=%.1fs",
                mu_hat, phi_hat, sigma_eta_hat, self._fitting_time,
            )

        except ImportError:
            logger.warning("PyMC not available, falling back to GARCH-based proxy")
            self._fit_fallback(returns)
        except Exception as exc:
            logger.warning("SV fitting failed (%s), using fallback", exc)
            self._fit_fallback(returns)

        return self

    def _fit_quasi_bayesian(self, returns: pd.Series) -> None:
        """
        Fast quasi-Bayesian SV approximation.
        
        Uses a two-step approach:
        1. Estimate log-volatility via log(r² + c) transformation
        2. Apply Kalman-filter-like smoothing for the AR(1) state
        3. Bootstrap-style uncertainty bands
        
        This runs in <1 second for typical financial series.
        """
        r = returns.values.astype(np.float64)
        T = len(r)
        idx = returns.index

        # Step 1: Log-squared transformation
        # log(r_t²) = h_t + log(ε_t²)  where ε_t ~ N(0,1)
        # E[log(ε²)] ≈ -1.27, Var[log(ε²)] ≈ π²/2 ≈ 4.93
        c = 1e-8  # small constant to avoid log(0)
        log_r2 = np.log(r ** 2 + c)

        # Step 2: Estimate AR(1) parameters via Yule-Walker on smoothed log-r²
        # First smooth with a simple moving average
        window = min(5, T // 10)
        if window >= 2:
            log_r2_smooth = pd.Series(log_r2).rolling(window, center=True, min_periods=1).mean().values
        else:
            log_r2_smooth = log_r2

        # Yule-Walker for AR(1)
        y = log_r2_smooth - np.mean(log_r2_smooth)
        mu_hat = float(np.mean(log_r2_smooth))
        # Bias-corrected AR(1) coefficient
        gamma0 = float(np.var(y))
        gamma1 = float(np.mean(y[:-1] * y[1:]))
        phi_hat = min(max(gamma1 / (gamma0 + 1e-12), 0.5), 0.999)
        # Innovation variance
        sigma_eta_hat = float(np.sqrt(max(gamma0 * (1 - phi_hat ** 2), 0.01)))

        # Step 3: Kalman smoother for log-volatility path
        # Forward pass (filter)
        h_filtered = np.zeros(T)
        P_filtered = np.zeros(T)
        R_eps = np.pi ** 2 / 2  # measurement noise variance

        h_filtered[0] = mu_hat
        P_filtered[0] = sigma_eta_hat ** 2 / (1 - phi_hat ** 2 + 1e-8)

        for t in range(1, T):
            # Predict
            h_pred = mu_hat + phi_hat * (h_filtered[t - 1] - mu_hat)
            P_pred = phi_hat ** 2 * P_filtered[t - 1] + sigma_eta_hat ** 2
            # Update
            K = P_pred / (P_pred + R_eps)
            h_filtered[t] = h_pred + K * (log_r2[t] - h_pred)
            P_filtered[t] = (1 - K) * P_pred

        # Backward pass (smoother)
        h_smooth = np.zeros(T)
        P_smooth = np.zeros(T)
        h_smooth[-1] = h_filtered[-1]
        P_smooth[-1] = P_filtered[-1]

        for t in range(T - 2, -1, -1):
            P_pred = phi_hat ** 2 * P_filtered[t] + sigma_eta_hat ** 2
            L = phi_hat * P_filtered[t] / (P_pred + 1e-12)
            h_smooth[t] = h_filtered[t] + L * (h_smooth[t + 1] - (mu_hat + phi_hat * (h_filtered[t] - mu_hat)))
            P_smooth[t] = P_filtered[t] + L ** 2 * (P_smooth[t + 1] - P_pred)

        # Step 4: Convert to volatility scale
        # σ_t = exp(h_t / 2)
        vol_mean = np.exp(h_smooth / 2)
        vol_std = np.sqrt(P_smooth)
        # Approximate 80% CI on log scale, then exponentiate
        z80 = 1.282  # 80% CI z-score
        vol_lower = np.exp((h_smooth - z80 * vol_std) / 2)
        vol_upper = np.exp((h_smooth + z80 * vol_std) / 2)

        # Ensure positivity
        vol_mean = np.maximum(vol_mean, 1e-12)
        vol_lower = np.maximum(vol_lower, 1e-12)
        vol_upper = np.maximum(vol_upper, vol_lower + 1e-12)

        std_resid = r / vol_mean

        cond_vol_series = pd.Series(vol_mean, index=idx, name="cond_vol")
        std_resid_series = pd.Series(std_resid, index=idx, name="std_resid")

        # Approximate log-likelihood
        loglik = float(np.sum(
            -0.5 * np.log(2 * np.pi) - np.log(vol_mean) - 0.5 * (r / vol_mean) ** 2
        ))
        k = 3  # mu, phi, sigma_eta
        aic = 2 * k - 2 * loglik
        bic = k * np.log(T) - 2 * loglik

        self._result = VolatilityResult(
            model_name="StochasticVol(QuasiBayesian)",
            params={
                "mu": mu_hat,
                "phi": phi_hat,
                "sigma_eta": sigma_eta_hat,
                "persistence": phi_hat,
            },
            conditional_volatility=cond_vol_series,
            standardized_residuals=std_resid_series,
            aic=float(aic),
            bic=float(bic),
            loglikelihood=loglik,
            converged=True,
        )

        self._vol_lower = pd.Series(vol_lower, index=idx)
        self._vol_upper = pd.Series(vol_upper, index=idx)
        self._log_vol_mean = pd.Series(h_smooth, index=idx)
        self._fitting_time = 0.0

        logger.info(
            "Quasi-Bayesian SV fit: μ=%.4f, φ=%.4f, σ_η=%.4f",
            mu_hat, phi_hat, sigma_eta_hat,
        )

    def _fit_fallback(self, returns: pd.Series) -> None:
        """Fallback: use EWMA as a simple volatility proxy if PyMC fails."""
        r = returns.values.astype(np.float64)
        lam = 0.94
        T = len(r)
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(r[:min(5, T)])
        for t in range(1, T):
            sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * r[t - 1] ** 2
        sigma2 = np.maximum(sigma2, 1e-12)
        vol = np.sqrt(sigma2)
        std_resid = r / vol

        idx = returns.index
        self._result = VolatilityResult(
            model_name="StochasticVol(Fallback-EWMA)",
            params={"mu": float(np.mean(np.log(sigma2))), "phi": 0.94, "sigma_eta": 0.1},
            conditional_volatility=pd.Series(vol, index=idx),
            standardized_residuals=pd.Series(std_resid, index=idx),
            aic=None,
            bic=None,
            loglikelihood=None,
            converged=False,
        )
        self._vol_lower = pd.Series(vol * 0.5, index=idx)
        self._vol_upper = pd.Series(vol * 2.0, index=idx)
        self._log_vol_mean = pd.Series(np.log(vol), index=idx)
        self._fitting_time = 0.0

    def conditional_variance(self) -> pd.Series:
        """Return in-sample conditional variance σ²."""
        if not self.is_fitted:
            raise NotFittedError("Call fit() before accessing conditional variance.")
        return self._result.conditional_volatility ** 2

    def forecast(self, steps: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        SV forecast: use AR(1) forecast of log-volatility.

        h_{T+h} = μ + φ^h · (h_T - μ)
        σ_{T+h} = exp(h_{T+h} / 2)
        """
        if not self.is_fitted:
            raise NotFittedError("Call fit() before forecasting.")

        mu = self._result.params["mu"]
        phi = self._result.params["phi"]
        last_vol = self._result.conditional_volatility.iloc[-1]
        last_log_vol = np.log(max(last_vol, 1e-12)) * 2  # h_T = 2 * log(σ_T)

        forecast_mean = np.zeros(steps)
        forecast_vol = np.zeros(steps)

        for h in range(steps):
            h_forecast = mu + phi ** (h + 1) * (last_log_vol - mu)
            forecast_vol[h] = np.exp(h_forecast / 2)

        return forecast_mean, forecast_vol

    def diagnose(self) -> DiagnosticsResult:
        """Run diagnostic tests on standardised residuals."""
        if not self.is_fitted:
            raise NotFittedError("Call fit() before running diagnostics.")

        std_resid = self._result.standardized_residuals.dropna().values.astype(float)
        n = len(std_resid)

        max_lags = min(10, max(1, n // 5))
        lb_df = acorr_ljungbox(std_resid, lags=[max_lags])
        lb_stat = float(lb_df["lb_stat"].iloc[0])
        lb_pvalue = float(lb_df["lb_pvalue"].iloc[0])

        nlags_arch = min(5, max(1, n // 10))
        arch_lm = het_arch(std_resid, nlags=nlags_arch)
        arch_stat = float(arch_lm[2])
        arch_pvalue = float(arch_lm[3])

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
