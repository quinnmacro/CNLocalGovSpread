"""
Model tournament: compare volatility models via AIC/BIC and diagnostics.

Provides a uniform comparison framework for heterogeneous models
(GARCH, FIGARCH, EWMA, ML) and selects the best performer.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.core.base import VolatilityModel
from src.core.logging_config import get_logger
from src.core.types import DiagnosticsResult

logger = get_logger(__name__)


class ModelTournament:
    """
    Tournament comparing multiple fitted VolatilityModel instances.

    Usage:
        t = ModelTournament()
        t.add_model("GARCH", garch_model)
        t.add_model("EWMA", ewma_model)
        results = t.run()
        winner = t.winner("aic")
    """

    def __init__(self) -> None:
        self._models: dict[str, VolatilityModel] = {}
        self._results: Optional[pd.DataFrame] = None

    def add_model(self, name: str, model: VolatilityModel) -> "ModelTournament":
        """Add a fitted VolatilityModel to the tournament. Returns self for chaining."""
        if not model.is_fitted:
            raise ValueError(f"Model '{name}' is not fitted. Call model.fit() first.")
        self._models[name] = model
        self._results = None  # invalidate cache
        return self

    def run(self) -> pd.DataFrame:
        """
        Run the tournament and return a DataFrame of model metrics.

        Columns: model_name, aic, bic, converged, persistence,
                 has_arch_effects, is_normal, lb_pvalue, arch_pvalue, jb_pvalue
        """
        rows = []
        for name, model in self._models.items():
            result = model.result
            diag = result.diagnostics

            row: dict = {
                "model_name": name,
                "model_type": result.model_name,
                "aic": result.aic,
                "bic": result.bic,
                "converged": result.converged,
                "persistence": result.persistence,
            }

            if diag is not None:
                row.update({
                    "has_arch_effects": diag.has_arch_effects,
                    "is_normal": diag.is_normal,
                    "lb_pvalue": diag.ljung_box_pvalue,
                    "arch_pvalue": diag.arch_lm_pvalue,
                    "jb_pvalue": diag.jarque_bera_pvalue,
                })
            else:
                row.update({
                    "has_arch_effects": None,
                    "is_normal": None,
                    "lb_pvalue": None,
                    "arch_pvalue": None,
                    "jb_pvalue": None,
                })

            # ML-specific metrics
            if result.aic is None:
                row["rmse_oos"] = result.params.get("rmse_oos")
                row["mae_oos"] = result.params.get("mae_oos")
            else:
                row["rmse_oos"] = None
                row["mae_oos"] = None

            rows.append(row)

        df = pd.DataFrame(rows).set_index("model_name")
        self._results = df

        logger.info("Tournament completed: %d models compared", len(df))
        return df

    def winner(self, criterion: str = "aic") -> str:
        """
        Select the winning model by criterion.

        Parameters
        ----------
        criterion : str
            "aic" or "bic" for information criteria (lower is better).
            "rmse" for ML models (lower is better).

        Returns
        -------
        str : Name of the winning model.
        """
        if self._results is None:
            self.run()

        df = self._results
        assert df is not None

        if criterion in ("aic", "bic"):
            # Only consider converged models with valid criterion
            valid = df[df["converged"] & df[criterion].notna()]
            if valid.empty:
                raise ValueError(f"No converged models with valid {criterion.upper()}")
            best_idx = valid[criterion].idxmin()
            return str(best_idx)

        elif criterion == "rmse":
            valid = df[df["rmse_oos"].notna()]
            if valid.empty:
                raise ValueError("No models with valid RMSE")
            best_idx = valid["rmse_oos"].idxmin()
            return str(best_idx)

        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def rank(self, criterion: str = "aic") -> list[str]:
        """Return model names sorted by criterion (best first)."""
        if self._results is None:
            self.run()

        df = self._results
        assert df is not None

        if criterion in ("aic", "bic"):
            valid = df[df["converged"] & df[criterion].notna()]
            return valid[criterion].sort_values().index.tolist()
        elif criterion == "rmse":
            valid = df[df["rmse_oos"].notna()]
            return valid["rmse_oos"].sort_values().index.tolist()
        return []

    def summary(self) -> str:
        """Pretty-print tournament results."""
        if self._results is None:
            self.run()

        df = self._results
        assert df is not None

        lines = [
            "=" * 80,
            "MODEL TOURNAMENT RESULTS",
            "=" * 80,
        ]

        for idx, row in df.iterrows():
            lines.append(f"\n  {idx} ({row['model_type']}):")
            if row["aic"] is not None:
                lines.append(f"    AIC={row['aic']:.2f}, BIC={row['bic']:.2f}")
            if row.get("rmse_oos") is not None:
                lines.append(f"    RMSE={row['rmse_oos']:.4f}, MAE={row['mae_oos']:.4f}")
            lines.append(f"    Converged: {row['converged']}")
            if row["persistence"] is not None:
                lines.append(f"    Persistence (α+β): {row['persistence']:.4f}")
            if row["has_arch_effects"] is not None:
                arch = "Yes ⚠️" if row["has_arch_effects"] else "No ✓"
                normal = "Yes ✓" if row["is_normal"] else "No ⚠️"
                lines.append(f"    ARCH effects: {arch}")
                lines.append(f"    Normality: {normal}")

        try:
            w = self.winner()
            lines.append(f"\n  Winner (AIC): {w}")
        except ValueError:
            pass

        lines.append("=" * 80)
        return "\n".join(lines)
