"""
Machine learning volatility models (XGBoost / LightGBM).

CRITICAL DESIGN DECISIONS:
- Target: r²[t+1] (next-period squared return) — NOT r²[t] (prevents leakage!)
- Features: lagged squared returns r²[t-k] for k=1..n_lags,
            plus rolling realised volatilities (5d, 20d, 60d windows)
- NO current-period return in features (would be look-ahead bias)
- Walk-forward training: train on [0:t], predict t+1, then roll forward
- Normalisation: StandardScaler fit on training fold ONLY
- Evaluation: RMSE and MAE (NOT AIC/BIC — these are not MLE models)
"""

from __future__ import annotations

from typing import Literal, Self

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from src.core.base import NotFittedError, VolatilityModel
from src.core.logging_config import get_logger
from src.core.types import DiagnosticsResult, VolatilityResult

logger = get_logger(__name__)

ModelType = Literal["xgboost", "lightgbm"]


class MLVolatilityModel(VolatilityModel):
    """
    Machine learning volatility model (XGBoost / LightGBM).

    Parameters
    ----------
    model_type : "xgboost" or "lightgbm"
    n_lags : Number of lagged squared-return features.
    n_cv_splits : Number of time-series cross-validation folds.
    """

    def __init__(
        self,
        model_type: ModelType = "xgboost",
        n_lags: int = 10,
        n_cv_splits: int = 5,
    ) -> None:
        valid_types = ("xgboost", "lightgbm")
        if model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}, got '{model_type}'")
        if n_lags < 1:
            raise ValueError(f"n_lags must be >= 1, got {n_lags}")

        self.model_type: ModelType = model_type
        self.n_lags = n_lags
        self.n_cv_splits = n_cv_splits

        self._model = None
        self._scaler: StandardScaler | None = None
        self._feature_names: list[str] = []
        self._returns: np.ndarray | None = None
        self._last_features: np.ndarray | None = None

        logger.debug("MLVolatilityModel: type=%s, n_lags=%d", model_type, n_lags)

    # ------------------------------------------------------------------
    # VolatilityModel interface
    # ------------------------------------------------------------------

    def fit(self, returns: pd.Series) -> Self:
        """
        Fit ML volatility model using walk-forward cross-validation.
        """
        if len(returns) < self.n_lags + 100:
            raise ValueError(
                f"ML model requires at least {self.n_lags + 100} observations, "
                f"got {len(returns)}"
            )

        self._returns = returns.values.astype(np.float64)
        r = self._returns

        X, y, valid_idx = self._build_features_and_target(r)
        self._feature_names = list(X.columns)

        logger.info(
            "Built %d features x %d samples for %s model",
            X.shape[1], X.shape[0], self.model_type,
        )

        cv_rmse = self._walk_forward_cv(X, y)
        logger.info("Walk-forward CV RMSE: %.6f", cv_rmse)

        # Final fit on all data
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X.values)
        self._model = self._create_model()
        self._model.fit(X_scaled, y.values)

        # In-sample predictions
        y_pred = self._model.predict(X_scaled)
        cond_var = np.maximum(y_pred, 1e-12)
        cond_vol = np.sqrt(cond_var)

        r_valid = r[valid_idx]
        std_resid = r_valid / cond_vol

        orig_idx = returns.index[valid_idx]
        cond_vol_series = pd.Series(cond_vol, index=orig_idx, name="cond_vol")
        std_resid_series = pd.Series(std_resid, index=orig_idx, name="std_resid")

        self._last_features = X.iloc[-1:].values

        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:5]
            top_features = [(self._feature_names[i], importances[i]) for i in top_idx]
            logger.info("Top 5 features: %s", top_features)

        self._result = VolatilityResult(
            model_name=f"ML_{self.model_type.upper()}(lags={self.n_lags})",
            params={"cv_rmse": cv_rmse, "n_features": X.shape[1]},
            conditional_volatility=cond_vol_series,
            standardized_residuals=std_resid_series,
            aic=None,
            bic=None,
            loglikelihood=None,
            converged=True,
        )

        logger.info(
            "%s fit complete: %d features, CV RMSE=%.6f",
            self.model_type.upper(), X.shape[1], cv_rmse,
        )
        return self

    def conditional_variance(self) -> pd.Series:
        """Return in-sample conditional variance σ²."""
        if not self.is_fitted:
            raise NotFittedError("Call fit() before accessing conditional variance.")
        return self._result.conditional_volatility ** 2

    def forecast(self, steps: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        Forecast conditional volatility iteratively: predict t+1,
        use that prediction as a feature to predict t+2, etc.
        """
        if not self.is_fitted:
            raise NotFittedError("Call fit() before forecasting.")

        assert self._scaler is not None
        assert self._model is not None
        assert self._last_features is not None

        forecast_vol = np.zeros(steps)
        forecast_mean = np.zeros(steps)

        current_features = self._last_features.copy()

        for h in range(steps):
            X_scaled = self._scaler.transform(current_features)
            pred_var = float(self._model.predict(X_scaled)[0])
            pred_var = max(pred_var, 1e-12)
            forecast_vol[h] = np.sqrt(pred_var)

            new_r2 = pred_var
            current_features = self._shift_features(current_features, new_r2)

        return forecast_mean, forecast_vol

    def diagnose(self) -> DiagnosticsResult:
        """Run diagnostic tests on prediction residuals."""
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
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_features_and_target(
        self, returns: np.ndarray
    ) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Build feature matrix X and target vector y with proper lag structure.

        Target: y[t] = r²[t+1]  (next period's squared return)
        Features at time t:
        - r²[t-1], r²[t-2], ..., r²[t-n_lags]
        - rolling_var_5[t], rolling_var_20[t], rolling_var_60[t]
        """
        T = len(returns)
        r2 = returns ** 2

        rv_5 = pd.Series(r2).rolling(5, min_periods=3).mean().values
        rv_20 = pd.Series(r2).rolling(20, min_periods=10).mean().values
        rv_60 = pd.Series(r2).rolling(60, min_periods=30).mean().values

        feature_dict: dict[str, np.ndarray] = {}

        for k in range(1, self.n_lags + 1):
            lagged = np.zeros(T)
            lagged[k:] = r2[:-k]
            feature_dict[f"r2_lag{k}"] = lagged

        feature_dict["rv_5"] = rv_5
        feature_dict["rv_20"] = rv_20
        feature_dict["rv_60"] = rv_60

        target = np.zeros(T)
        target[:-1] = r2[1:]
        target[-1] = np.nan

        min_warmup = max(self.n_lags, 60)
        valid_mask = np.ones(T, dtype=bool)
        valid_mask[:min_warmup] = False
        valid_mask[-1] = False

        for feat_arr in feature_dict.values():
            valid_mask &= ~np.isnan(feat_arr)
        valid_mask &= ~np.isnan(target)

        valid_idx = np.where(valid_mask)[0]

        X = pd.DataFrame(
            {k: v[valid_idx] for k, v in feature_dict.items()},
            index=valid_idx,
        )
        y = pd.Series(target[valid_idx], index=valid_idx, name="target_r2")

        return X, y, valid_idx

    def _shift_features(self, features: np.ndarray, new_r2: float) -> np.ndarray:
        """Shift features forward by one step for iterative forecasting."""
        new_features = features.copy()
        n_lags = self.n_lags

        for k in range(n_lags - 1, 0, -1):
            new_features[0, k] = new_features[0, k - 1]
        new_features[0, 0] = new_r2

        for i, w in enumerate([5, 20, 60]):
            idx = n_lags + i
            new_features[0, idx] = (1 - 1.0 / w) * new_features[0, idx] + new_r2 / w

        return new_features

    # ------------------------------------------------------------------
    # Walk-forward CV
    # ------------------------------------------------------------------

    def _walk_forward_cv(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Walk-forward cross-validation using TimeSeriesSplit."""
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        rmse_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train = X.iloc[train_idx].values
            y_train = y.iloc[train_idx].values
            X_test = X.iloc[test_idx].values
            y_test = y.iloc[test_idx].values

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = self._create_model()
            model.fit(X_train_scaled, y_train)
            y_pred = np.maximum(model.predict(X_test_scaled), 0.0)

            rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
            rmse_scores.append(rmse)

            logger.debug(
                "CV fold %d: RMSE=%.6f (%d train, %d test)",
                fold_idx, rmse, len(train_idx), len(test_idx),
            )

        return float(np.mean(rmse_scores))

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------

    def _create_model(self):
        """Create a fresh ML model instance with default hyperparameters."""
        if self.model_type == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise ImportError(
                    "XGBoost not installed. Install with: pip install xgboost"
                ) from exc

            return XGBRegressor(
                max_depth=4,
                learning_rate=0.05,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=42,
                verbosity=0,
            )
        elif self.model_type == "lightgbm":
            try:
                from lightgbm import LGBMRegressor
            except ImportError as exc:
                raise ImportError(
                    "LightGBM not installed. Install with: pip install lightgbm"
                ) from exc

            return LGBMRegressor(
                max_depth=4,
                learning_rate=0.05,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="regression",
                random_state=42,
                verbose=-1,
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
