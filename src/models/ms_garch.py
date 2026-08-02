"""
Markov-Switching GARCH (MS-GARCH) model.

Combines GARCH volatility dynamics with regime switching:
    r_t | S_t=k ~ N(0, σ²_{k,t})
    σ²_{k,t} = ω_k + α_k · r²_{t-1} + β_k · σ²_{k,t-1}

where S_t ∈ {1, ..., K} is a hidden Markov chain with transition matrix Π.

The key insight over standard GARCH: volatility clustering is driven not
just by a single AR(1)-like process, but by switches between qualitatively
different regimes (e.g., "calm" with low ω, α, β vs. "turbulent" with high
values). This bridges the regimes page (HMM state detection) with the
volatility page (GARCH dynamics).

Implementation:
    1. Fit GARCH(1,1) on the full series as baseline
    2. Fit HMM on standardized residuals to detect regimes
    3. Re-estimate regime-specific GARCH parameters via EM-like procedure
    4. Combine using filtered regime probabilities

Simplified approach for robustness:
    - Step 1: Fit standard GARCH(1,1) for initial σ²_t
    - Step 2: Use HMM on |r_t| (absolute returns) to identify K regimes
    - Step 3: For each regime, estimate GARCH params using regime-weighted MLE
    - Step 4: Conditional volatility uses filtered regime probabilities
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, norm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from src.core.base import NotFittedError, VolatilityModel
from src.core.logging_config import get_logger
from src.core.types import DiagnosticsResult, VolatilityResult

logger = get_logger(__name__)


class MSGARCHModel(VolatilityModel):
    """
    Markov-Switching GARCH model.

    Parameters
    ----------
    n_regimes : int
        Number of regimes (default 2: calm + turbulent).
    """

    def __init__(self, n_regimes: int = 2) -> None:
        if n_regimes < 2:
            raise ValueError(f"n_regimes must be >= 2, got {n_regimes}")
        self.n_regimes = n_regimes
        self._returns: np.ndarray | None = None

        logger.debug("MSGARCHModel created: n_regimes=%d", n_regimes)

    # ------------------------------------------------------------------
    # VolatilityModel interface
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series) -> Self:
        """Fit MS-GARCH: GARCH + HMM + regime-specific parameters."""
        if len(returns) < 50:
            raise ValueError(f"MS-GARCH requires at least 50 observations, got {len(returns)}")

        self._returns = returns.values.astype(np.float64)
        r = self._returns
        T = len(r)
        idx = returns.index

        try:
            from arch import arch_model  # type: ignore[import-untyped]
            from hmmlearn.hmm import GaussianHMM

            # Step 1: Fit baseline GARCH(1,1) with Student-t
            base_model = arch_model(
                returns, mean="Constant", vol="GARCH",
                p=1, o=0, q=1, dist="studentst",
            )
            base_fit = base_model.fit(disp="off", show_warning=False)
            base_vol = base_fit.conditional_volatility.dropna().values.astype(float)
            base_std_resid = base_fit.std_resid.dropna().values.astype(float)

            # Step 2: Fit HMM on absolute returns to detect volatility regimes
            # Use |r_t| as the observation — this captures volatility regimes
            obs = np.abs(r).reshape(-1, 1)

            # Ensure we have enough observations
            min_obs = max(50, T // 10)
            if T < min_obs:
                raise ValueError("Insufficient data for HMM fitting")

            hmm = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=42,
            )
            hmm.fit(obs)
            regime_labels = hmm.predict(obs)
            regime_probs = hmm.predict_proba(obs)

            # Sort regimes by mean volatility (ascending)
            regime_means = np.array([
                np.mean(np.abs(r[regime_labels == k]))
                for k in range(self.n_regimes)
            ])
            sort_order = np.argsort(regime_means)

            # Relabel regimes
            label_map = {old: new for new, old in enumerate(sort_order)}
            regime_labels_sorted = np.array([label_map[l] for l in regime_labels])
            regime_probs_sorted = regime_probs[:, sort_order]

            # Step 3: Estimate regime-specific GARCH parameters
            regime_params = []
            regime_vol_series = np.zeros((T, self.n_regimes))

            for k in range(self.n_regimes):
                mask = regime_labels_sorted == k
                n_k = mask.sum()

                if n_k < 30:
                    # Too few observations — use scaled baseline
                    scale = regime_means[sort_order[k]] / (np.mean(np.abs(r)) + 1e-12)
                    regime_vol_series[:, k] = base_vol * scale
                    regime_params.append({
                        "omega": float(base_fit.params.get("omega", 1e-6)) * scale ** 2,
                        "alpha": float(base_fit.params.get("alpha[1]", 0.1)),
                        "beta": float(base_fit.params.get("beta[1]", 0.85)),
                        "regime_mean_abs": float(regime_means[sort_order[k]]),
                    })
                    continue

                # Fit GARCH on regime-weighted observations
                r_k = returns.copy()
                # Weight by regime probability for smoother estimation
                weights = regime_probs_sorted[:, k]

                try:
                    # Use regime-specific data with soft weighting
                    weighted_r = r_k * np.sqrt(weights)
                    wm = arch_model(
                        weighted_r, mean="Constant", vol="GARCH",
                        p=1, o=0, q=1, dist="normal",
                    )
                    wf = wm.fit(disp="off", show_warning=False)

                    omega_k = float(wf.params.get("omega", 1e-6))
                    alpha_k = float(wf.params.get("alpha[1]", 0.1))
                    beta_k = float(wf.params.get("beta[1]", 0.85))

                    # Ensure valid parameters
                    alpha_k = np.clip(alpha_k, 0.01, 0.4)
                    beta_k = np.clip(beta_k, 0.5, 0.99)
                    if alpha_k + beta_k >= 0.999:
                        scale = 0.999 / (alpha_k + beta_k)
                        alpha_k *= scale
                        beta_k *= scale

                except Exception:
                    # Fallback: use scaled baseline
                    scale = regime_means[sort_order[k]] / (np.mean(np.abs(r)) + 1e-12)
                    omega_k = float(base_fit.params.get("omega", 1e-6)) * scale ** 2
                    alpha_k = float(base_fit.params.get("alpha[1]", 0.1))
                    beta_k = float(base_fit.params.get("beta[1]", 0.85))

                # Compute regime-specific volatility path
                f_k = np.full(T, np.var(r))
                for t in range(1, T):
                    f_k[t] = omega_k + alpha_k * r[t - 1] ** 2 + beta_k * f_k[t - 1]
                    f_k[t] = max(f_k[t], 1e-12)
                regime_vol_series[:, k] = np.sqrt(f_k)

                regime_params.append({
                    "omega": omega_k,
                    "alpha": alpha_k,
                    "beta": beta_k,
                    "regime_mean_abs": float(regime_means[sort_order[k]]),
                })

            # Step 4: Compute mixture volatility using filtered probabilities
            # σ²_t = Σ_k P(S_t=k) · σ²_{k,t}
            regime_var = regime_vol_series ** 2
            mixture_var = np.sum(regime_probs_sorted * regime_var, axis=1)
            mixture_var = np.maximum(mixture_var, 1e-12)
            cond_vol = np.sqrt(mixture_var)

            std_resid = r / cond_vol

            # Transition matrix
            trans_matrix = hmm.transmat_[np.ix_(sort_order, sort_order)]

            # Model statistics
            loglik = float(base_fit.loglikelihood)  # approximate
            k_total = 3 * self.n_regimes + self.n_regimes ** 2  # params + transition
            aic = 2 * k_total - 2 * loglik
            bic = k_total * np.log(T) - 2 * loglik

            cond_vol_series = pd.Series(cond_vol, index=idx, name="cond_vol")
            std_resid_series = pd.Series(std_resid, index=idx, name="std_resid")

            # Current regime
            current_regime = int(regime_labels_sorted[-1])
            regime_names = {0: "低波动", 1: "高波动", 2: "极端波动"}

            # Build flat params dict
            flat_params: dict[str, float] = {}
            for k_idx, rp in enumerate(regime_params):
                prefix = f"k{k_idx}_"
                flat_params[f"{prefix}omega"] = rp["omega"]
                flat_params[f"{prefix}alpha"] = rp["alpha"]
                flat_params[f"{prefix}beta"] = rp["beta"]
                flat_params[f"{prefix}mean_abs"] = rp["regime_mean_abs"]

            # Overall persistence (weighted average)
            avg_persistence = np.mean([
                rp["alpha"] + rp["beta"] for rp in regime_params
            ])
            flat_params["persistence"] = float(avg_persistence)
            flat_params["current_regime"] = float(current_regime)

            self._result = VolatilityResult(
                model_name=f"MS-GARCH({self.n_regimes} regimes)",
                params=flat_params,
                conditional_volatility=cond_vol_series,
                standardized_residuals=std_resid_series,
                aic=float(aic),
                bic=float(bic),
                loglikelihood=loglik,
                converged=True,
            )

            # Store extra info for visualization
            self._regime_labels = pd.Series(regime_labels_sorted, index=idx)
            self._regime_probs = pd.DataFrame(
                regime_probs_sorted, index=idx,
                columns=[f"prob_k{k}" for k in range(self.n_regimes)],
            )
            self._regime_params = regime_params
            self._trans_matrix = trans_matrix
            self._current_regime = current_regime
            self._current_regime_name = regime_names.get(
                current_regime, f"Regime-{current_regime}"
            )
            self._regime_vol_series = pd.DataFrame(
                regime_vol_series, index=idx,
                columns=[f"vol_k{k}" for k in range(self.n_regimes)],
            )

            logger.info(
                "MS-GARCH fit complete: %d regimes, current=%s, AIC=%.2f",
                self.n_regimes, self._current_regime_name, aic,
            )

        except ImportError as exc:
            logger.warning("Required library not available (%s), using fallback", exc)
            self._fit_fallback(returns)
        except Exception as exc:
            logger.warning("MS-GARCH fitting failed (%s), using fallback", exc)
            self._fit_fallback(returns)

        return self

    def _fit_fallback(self, returns: pd.Series) -> None:
        """Fallback: fit simple GARCH and label regimes by volatility level."""
        r = returns.values.astype(np.float64)
        T = len(r)
        idx = returns.index

        # Simple EWMA as fallback
        lam = 0.94
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(r[:min(5, T)])
        for t in range(1, T):
            sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * r[t - 1] ** 2
        sigma2 = np.maximum(sigma2, 1e-12)
        vol = np.sqrt(sigma2)
        std_resid = r / vol

        # Simple regime classification by volatility quantile
        vol_median = np.median(vol)
        labels = (vol > vol_median).astype(int)
        probs = np.column_stack([1 - (vol > vol_median).astype(float),
                                  (vol > vol_median).astype(float)])

        self._result = VolatilityResult(
            model_name="MS-GARCH(Fallback-EWMA)",
            params={"persistence": 0.94, "current_regime": float(labels[-1])},
            conditional_volatility=pd.Series(vol, index=idx),
            standardized_residuals=pd.Series(std_resid, index=idx),
            aic=None,
            bic=None,
            loglikelihood=None,
            converged=False,
        )

        self._regime_labels = pd.Series(labels, index=idx)
        self._regime_probs = pd.DataFrame(probs, index=idx, columns=["prob_k0", "prob_k1"])
        self._regime_params = [
            {"omega": 1e-6, "alpha": 0.06, "beta": 0.88, "regime_mean_abs": float(np.mean(np.abs(r[labels == 0])))},
            {"omega": 1e-6, "alpha": 0.12, "beta": 0.85, "regime_mean_abs": float(np.mean(np.abs(r[labels == 1])))},
        ]
        self._trans_matrix = np.array([[0.95, 0.05], [0.05, 0.95]])
        self._current_regime = int(labels[-1])
        self._current_regime_name = "低波动" if labels[-1] == 0 else "高波动"
        self._regime_vol_series = pd.DataFrame(
            np.column_stack([vol, vol * 1.5]),
            index=idx, columns=["vol_k0", "vol_k1"],
        )

    def conditional_variance(self) -> pd.Series:
        """Return in-sample conditional variance σ²."""
        if not self.is_fitted:
            raise NotFittedError("Call fit() before accessing conditional variance.")
        return self._result.conditional_volatility ** 2

    def forecast(self, steps: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """MS-GARCH forecast using regime-specific recursion + transition probs."""
        if not self.is_fitted:
            raise NotFittedError("Call fit() before forecasting.")

        r = self._returns
        assert r is not None

        last_r2 = r[-1] ** 2
        current_regime = self._current_regime

        forecast_mean = np.zeros(steps)
        forecast_vol = np.zeros(steps)

        # Use stationary distribution for long-horizon mixture
        eigvals, eigvecs = np.linalg.eig(self._trans_matrix.T)
        idx_max = np.argmin(np.abs(eigvals - 1.0))
        stat_dist = np.real(eigvecs[:, idx_max])
        stat_dist = stat_dist / stat_dist.sum()

        # Start from current regime
        regime_prob = np.zeros(self.n_regimes)
        regime_prob[current_regime] = 1.0

        # Get last volatility per regime
        last_f = {}
        for k in range(self.n_regimes):
            col = f"vol_k{k}"
            if hasattr(self, "_regime_vol_series") and col in self._regime_vol_series.columns:
                last_f[k] = self._regime_vol_series[col].iloc[-1] ** 2
            else:
                last_f[k] = self._result.conditional_volatility.iloc[-1] ** 2

        for h in range(steps):
            # Propagate regime probabilities
            regime_prob = regime_prob @ self._trans_matrix

            # Compute mixture variance
            mix_var = 0.0
            for k in range(self.n_regimes):
                rp = self._regime_params[k]
                f_k = rp["omega"] + rp["alpha"] * last_r2 + rp["beta"] * last_f[k]
                f_k = max(f_k, 1e-12)
                mix_var += regime_prob[k] * f_k
                last_f[k] = f_k

            forecast_vol[h] = np.sqrt(mix_var)

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
