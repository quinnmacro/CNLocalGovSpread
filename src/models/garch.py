"""
GARCH / EGARCH / GJR-GARCH volatility model wrapper around the `arch` library.

Supports:
- Standard GARCH(p, q) with normal / Student-t / skew-t innovations
- Exponential GARCH (EGARCH) — captures leverage effects
- GJR-GARCH (Glosten-Jagannathan-Runkle) — asymmetric response to shocks

All three share the same VolatilityModel interface so they can be compared
uniformly in the model selection tournament.
"""

from __future__ import annotations

from typing import Literal, Self

import numpy as np
import pandas as pd
from arch import arch_model  # type: ignore[import-untyped]
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from src.core.base import NotFittedError, VolatilityModel
from src.core.logging_config import get_logger
from src.core.types import DiagnosticsResult, VolatilityResult

logger = get_logger(__name__)

ModelType = Literal["garch", "egarch", "gjr"]


class GARCHModel(VolatilityModel):
    """
    Wrapper around the `arch` library for GARCH / EGARCH / GJR-GARCH models.

    Parameters
    ----------
    model_type : One of "garch", "egarch", "gjr"
    p : ARCH lag order (number of lagged squared-residual terms)
    q : GARCH lag order (number of lagged variance terms)
    dist : Innovation distribution — "normal", "studentst", "skewt"
    """

    def __init__(
        self,
        model_type: ModelType = "garch",
        p: int = 1,
        q: int = 1,
        dist: str = "studentst",
    ) -> None:
        valid_types = ("garch", "egarch", "gjr")
        if model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}, got \'{model_type}\'")
        valid_dists = ("normal", "studentst", "skewt")
        if dist not in valid_dists:
            raise ValueError(f"dist must be one of {valid_dists}, got \'{dist}\'")

        self.model_type: ModelType = model_type
        self.p = p
        self.q = q
        self.dist = dist

        self._model = None
        self._fit_result = None
        self._returns: pd.Series | None = None

        logger.debug("GARCHModel created: type=%s, p=%d, q=%d, dist=%s", model_type, p, q, dist)

    # ------------------------------------------------------------------
    # VolatilityModel interface
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series) -> Self:
        """Fit the GARCH-family model to a return series."""
        if returns.empty:
            raise ValueError("Cannot fit model on empty return series.")

        self._returns = returns.copy()

        if self.model_type == "garch":
            self._model = arch_model(
                returns, mean="Constant", vol="GARCH",
                p=self.p, o=0, q=self.q, dist=self.dist,
            )
        elif self.model_type == "gjr":
            self._model = arch_model(
                returns, mean="Constant", vol="GARCH",
                p=self.p, o=1, q=self.q, dist=self.dist,
            )
        elif self.model_type == "egarch":
            self._model = arch_model(
                returns, mean="Constant", vol="EGARCH",
                p=self.p, o=0, q=self.q, dist=self.dist,
            )

        logger.info(
            "Fitting %s(%d,%d) with dist=%s on %d observations...",
            self.model_type.upper(), self.p, self.q, self.dist, len(returns),
        )

        try:
            self._fit_result = self._model.fit(disp="off", show_warning=False)
        except Exception as exc:
            logger.error("GARCH fitting failed: %s", exc)
            raise

        convergence_ok = self._fit_result.convergence_flag == 0
        if not convergence_ok:
            logger.warning(
                "%s did not converge (flag=%d). Results may be unreliable.",
                self.model_type.upper(), self._fit_result.convergence_flag,
            )

        params = self._extract_params()
        cond_vol = self._fit_result.conditional_volatility.dropna()
        std_resid = self._fit_result.std_resid.dropna()

        self._result = VolatilityResult(
            model_name=f"{self.model_type.upper()}({self.p},{self.q})",
            params=params,
            conditional_volatility=cond_vol,
            standardized_residuals=std_resid,
            aic=float(self._fit_result.aic),
            bic=float(self._fit_result.bic),
            loglikelihood=float(self._fit_result.loglikelihood),
            converged=convergence_ok,
        )

        logger.info(
            "%s fit complete: AIC=%.2f, BIC=%.2f, converged=%s",
            self.model_type.upper(),
            self._result.aic, self._result.bic, convergence_ok,
        )
        return self

    def conditional_variance(self) -> pd.Series:
        """Return the in-sample conditional variance σ²."""
        if not self.is_fitted:
            raise NotFittedError("Call fit() before accessing conditional variance.")
        return self._result.conditional_volatility ** 2

    def forecast(self, steps: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        Forecast conditional mean and volatility for `steps` periods ahead.

        For EGARCH, analytic forecasts are only available for horizon=1,
        so we use simulation-based forecasts for longer horizons.

        Returns
        -------
        forecast_mean : ndarray of shape (steps,)
        forecast_vol  : ndarray of shape (steps,) — σ (not σ²)
        """
        if not self.is_fitted:
            raise NotFittedError("Call fit() before forecasting.")

        try:
            fcast = self._fit_result.forecast(horizon=steps)
        except ValueError:
            # EGARCH analytic forecasts unavailable for horizon > 1
            # Use simulation-based forecasts instead
            fcast = self._fit_result.forecast(
                horizon=steps, method="simulation", simulations=1000
            )

        mean_fcast = fcast.mean.iloc[-1].values.astype(float)
        var_fcast = fcast.variance.iloc[-1].values.astype(float)
        vol_fcast = np.sqrt(np.maximum(var_fcast, 0.0))

        return mean_fcast, vol_fcast

    def diagnose(self) -> DiagnosticsResult:
        """
        Run diagnostic tests on standardised residuals:
        - Ljung-Box(10) for remaining autocorrelation
        - ARCH-LM(5) for remaining ARCH effects
        - Jarque-Bera for non-normality
        """
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
        arch_stat = float(arch_lm[2])  # F-statistic
        arch_pvalue = float(arch_lm[3])  # F-pvalue

        # Jarque-Bera
        jb_res = jarque_bera(std_resid)
        jb_stat = float(jb_res.statistic)
        jb_pvalue = float(jb_res.pvalue)

        diag = DiagnosticsResult(
            ljung_box_stat=lb_stat,
            ljung_box_pvalue=lb_pvalue,
            arch_lm_stat=arch_stat,
            arch_lm_pvalue=arch_pvalue,
            jarque_bera_stat=jb_stat,
            jarque_bera_pvalue=jb_pvalue,
            n_obs=n,
        )

        logger.info("Diagnostics for %s:\n%s", self._result.model_name, diag.summary())
        return diag

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_params(self) -> dict[str, float]:
        """Extract fitted parameters into a flat dict with normalised names."""
        params: dict[str, float] = {}
        for name, value in self._fit_result.params.items():
            raw = name.lower().strip()
            # Normalise arch parameter names for VolatilityResult.persistence
            if raw in ("mu", "omega"):
                key = raw
            elif raw.startswith("alpha"):
                key = "alpha"  # "alpha[1]" -> "alpha"
            elif raw.startswith("beta"):
                key = "beta"   # "beta[1]" -> "beta"
            elif raw.startswith("gamma"):
                key = "gamma"  # GJR asymmetry term
            elif raw in ("nu", "df"):
                key = "nu"     # Student-t degrees of freedom
            else:
                key = raw.replace(" ", "_").replace("[", "").replace("]", "")
            params[key] = float(value)
        return params
