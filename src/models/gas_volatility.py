"""
Generalized Autoregressive Score (GAS) volatility model (Creal, Koopman, Lucas 2013).

Also known as "score-driven" or "observation-driven" models. GAS generalizes
GARCH by using the scaled score of the predictive distribution as the driving
force for time-varying parameters.

GAS(1,1) for volatility with Student-t distribution:

    f_t = σ²_t (time-varying variance)
    r_t ~ Student-t(ν, 0, f_t)

    s_t = score of log-likelihood w.r.t. f_t
        = [(ν+1)·r²_t / ((ν-2)·f_t + r²_t) - 1] / (2·f_t)  (for Student-t)

    f_{t+1} = ω + A·s_t + B·f_t

For Normal distribution, the score simplifies and GAS(1,1) becomes
exactly GARCH(1,1):
    s_t = (r²_t - f_t) / (2·f_t²)
    f_{t+1} = ω + α·r²_t + β·f_t

Key advantage: with heavy-tailed distributions (Student-t), the GAS
update automatically down-weights extreme observations, making it more
robust than GARCH to outliers.

Estimation: QMLE via scipy.optimize.minimize (L-BFGS-B).
"""

from __future__ import annotations

from typing import Literal, Self

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import digamma, gammaln
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from src.core.base import NotFittedError, VolatilityModel
from src.core.logging_config import get_logger
from src.core.types import DiagnosticsResult, VolatilityResult

logger = get_logger(__name__)

DistType = Literal["normal", "studentst"]


class GASVolModel(VolatilityModel):
    """
    GAS(1,1) volatility model with Normal or Student-t distribution.

    Parameters
    ----------
    dist : "normal" or "studentst"
        Predictive distribution. "normal" recovers GARCH(1,1).
    """

    def __init__(self, dist: DistType = "studentst") -> None:
        if dist not in ("normal", "studentst"):
            raise ValueError(f"dist must be 'normal' or 'studentst', got '{dist}'")
        self.dist = dist
        self._returns: np.ndarray | None = None

        logger.debug("GASVolModel created: dist=%s", dist)

    # ------------------------------------------------------------------
    # VolatilityModel interface
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series) -> Self:
        """Fit GAS(1,1) via QMLE."""
        if len(returns) < 30:
            raise ValueError(f"GAS requires at least 30 observations, got {len(returns)}")

        self._returns = returns.values.astype(np.float64)
        r = self._returns
        T = len(r)

        if self.dist == "normal":
            params, loglik, f_series = self._fit_normal(r)
        else:
            params, loglik, f_series = self._fit_studentt(r)

        cond_vol = np.sqrt(np.maximum(f_series, 1e-12))
        std_resid = r / cond_vol

        idx = returns.index
        cond_vol_series = pd.Series(cond_vol, index=idx, name="cond_vol")
        std_resid_series = pd.Series(std_resid, index=idx, name="std_resid")

        k = len(params)
        aic = 2 * k - 2 * loglik
        bic = k * np.log(T) - 2 * loglik

        param_dict = dict(params)
        if self.dist == "studentst":
            param_dict["persistence"] = float(params.get("B", 0))
        else:
            param_dict["persistence"] = float(params.get("alpha", 0) + params.get("beta", 0))

        self._result = VolatilityResult(
            model_name=f"GAS(1,1)-{self.dist.capitalize()}",
            params=param_dict,
            conditional_volatility=cond_vol_series,
            standardized_residuals=std_resid_series,
            aic=float(aic),
            bic=float(bic),
            loglikelihood=float(loglik),
            converged=True,
        )

        # Store score series for visualization
        self._score_series = pd.Series(
            self._compute_scores(r, f_series, params),
            index=idx,
        )
        self._f_series = pd.Series(f_series, index=idx)

        logger.info(
            "GAS fit complete: dist=%s, AIC=%.2f, BIC=%.2f, params=%s",
            self.dist, aic, bic, params,
        )
        return self

    def conditional_variance(self) -> pd.Series:
        """Return in-sample conditional variance σ²."""
        if not self.is_fitted:
            raise NotFittedError("Call fit() before accessing conditional variance.")
        return self._result.conditional_volatility ** 2

    def forecast(self, steps: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """GAS forecast: iterate the filtering equation forward."""
        if not self.is_fitted:
            raise NotFittedError("Call fit() before forecasting.")

        params = self._result.params
        last_f = self._f_series.iloc[-1]

        forecast_mean = np.zeros(steps)
        forecast_vol = np.zeros(steps)

        if self.dist == "normal":
            omega = params.get("omega", 0)
            alpha = params.get("alpha", 0)
            beta = params.get("beta", 0)
            # At forecast origin, expected score is 0
            # f_{T+h} = ω + (α + β) · f_{T+h-1}
            f = last_f
            for h in range(steps):
                f = omega + (alpha + beta) * f
                f = max(f, 1e-12)
                forecast_vol[h] = np.sqrt(f)
        else:
            omega = params.get("omega", 0)
            A = params.get("A", 0)
            B = params.get("B", 0)
            # Expected score = 0 under correct specification
            # f_{T+h} = ω + B · f_{T+h-1}
            f = last_f
            for h in range(steps):
                f = omega + B * f
                f = max(f, 1e-12)
                forecast_vol[h] = np.sqrt(f)

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

    # ------------------------------------------------------------------
    # Internal: Normal distribution (recovers GARCH)
    # ------------------------------------------------------------------

    def _fit_normal(
        self, r: np.ndarray
    ) -> tuple[dict[str, float], float, np.ndarray]:
        """Fit GAS with Normal distribution (= GARCH(1,1) via QMLE)."""
        T = len(r)
        f_init = np.var(r)

        def neg_loglik(theta: np.ndarray) -> float:
            omega, alpha, beta = theta
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
                return 1e12

            f = np.full(T, f_init)
            for t in range(1, T):
                f[t] = omega + alpha * r[t - 1] ** 2 + beta * f[t - 1]
                f[t] = max(f[t], 1e-12)

            ll = -0.5 * np.sum(
                np.log(2 * np.pi) + np.log(f) + r ** 2 / f
            )
            return -ll

        # Initial guess
        theta0 = np.array([
            f_init * 0.01,  # omega
            0.08,           # alpha
            0.88,           # beta
        ])

        bounds = [
            (1e-8, 1.0),   # omega
            (0.0, 0.5),    # alpha
            (0.5, 0.999),  # beta
        ]

        result = optimize.minimize(
            neg_loglik, theta0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        omega, alpha, beta = result.x
        # Recompute f series
        f = np.full(T, f_init)
        for t in range(1, T):
            f[t] = omega + alpha * r[t - 1] ** 2 + beta * f[t - 1]
            f[t] = max(f[t], 1e-12)

        ll = -float(neg_loglik(result.x))

        params = {
            "omega": float(omega),
            "alpha": float(alpha),
            "beta": float(beta),
        }
        return params, ll, f

    # ------------------------------------------------------------------
    # Internal: Student-t distribution
    # ------------------------------------------------------------------

    def _fit_studentt(
        self, r: np.ndarray
    ) -> tuple[dict[str, float], float, np.ndarray]:
        """Fit GAS with Student-t distribution (score-driven update)."""
        T = len(r)
        f_init = np.var(r)

        def neg_loglik(theta: np.ndarray) -> float:
            omega, A, B, nu = theta
            if omega <= 0 or B < 0 or B >= 0.9999 or nu <= 2.1:
                return 1e12

            f = np.full(T, f_init)
            for t in range(1, T):
                # Score for Student-t
                score_t = self._student_score(r[t - 1], f[t - 1], nu)
                f[t] = omega + A * score_t + B * f[t - 1]
                f[t] = max(f[t], 1e-12)

            # Log-likelihood for Student-t
            ll = 0.0
            for t in range(T):
                ll += self._student_logpdf(r[t], f[t], nu)

            return -ll

        theta0 = np.array([
            f_init * 0.01,  # omega
            0.5,            # A
            0.90,           # B
            6.0,            # nu (degrees of freedom)
        ])

        bounds = [
            (1e-8, 1.0),    # omega
            (0.0, 5.0),     # A
            (0.5, 0.999),   # B
            (2.1, 30.0),    # nu
        ]

        result = optimize.minimize(
            neg_loglik, theta0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        omega, A, B, nu = result.x
        # Recompute f series
        f = np.full(T, f_init)
        for t in range(1, T):
            score_t = self._student_score(r[t - 1], f[t - 1], nu)
            f[t] = omega + A * score_t + B * f[t - 1]
            f[t] = max(f[t], 1e-12)

        ll = -float(neg_loglik(result.x))

        params = {
            "omega": float(omega),
            "A": float(A),
            "B": float(B),
            "nu": float(nu),
        }
        return params, ll, f

    # ------------------------------------------------------------------
    # Score and log-pdf helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _student_score(r: float, f: float, nu: float) -> float:
        """
        Score of Student-t log-likelihood w.r.t. f (variance parameter).

        s_t = ∂ℓ/∂f = [(ν+1)·r² / ((ν-2)·f + r²) - 1] / (2·f)
        """
        f = max(f, 1e-12)
        numerator = (nu + 1) * r ** 2 / ((nu - 2) * f + r ** 2) - 1
        return numerator / (2 * f)

    @staticmethod
    def _student_logpdf(r: float, f: float, nu: float) -> float:
        """Log-pdf of Student-t with variance f and df nu."""
        f = max(f, 1e-12)
        # Student-t with location 0, scale σ = sqrt(f * (ν-2)/ν)
        # ℓ = log Γ((ν+1)/2) - log Γ(ν/2) - 0.5 log(νπ) - 0.5 log(σ²)
        #     - (ν+1)/2 · log(1 + r²/(ν·σ²))
        sigma2 = f * (nu - 2) / nu
        sigma2 = max(sigma2, 1e-12)

        return float(
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi)
            - 0.5 * np.log(sigma2)
            - (nu + 1) / 2 * np.log(1 + r ** 2 / (nu * sigma2))
        )

    def _compute_scores(
        self, r: np.ndarray, f: np.ndarray, params: dict
    ) -> np.ndarray:
        """Compute the score series for visualization."""
        T = len(r)
        scores = np.zeros(T)
        if self.dist == "normal":
            for t in range(T):
                ft = max(f[t], 1e-12)
                scores[t] = (r[t] ** 2 - ft) / (2 * ft ** 2)
        else:
            nu = params.get("nu", 6.0)
            for t in range(T):
                scores[t] = self._student_score(r[t], f[t], nu)
        return scores
