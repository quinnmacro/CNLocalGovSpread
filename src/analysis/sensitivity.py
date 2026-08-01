"""
Parameter sensitivity analysis.

One-at-a-time (OAT) perturbation and tornado diagrams for understanding
how model outputs depend on input parameters.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.core.logging_config import get_logger

logger = get_logger(__name__)


class SensitivityAnalyzer:
    """
    Parameter sensitivity analysis toolkit.

    Supports one-at-a-time (OAT) perturbation, tornado diagrams,
    and local elasticity computation.
    """

    def __init__(self, base_params: dict[str, float]) -> None:
        self._base = dict(base_params)

    def one_at_a_time(
        self,
        param_name: str,
        values: np.ndarray | list[float],
        metric_fn: Callable[[dict[str, float]], float],
    ) -> pd.DataFrame:
        """
        Vary one parameter while holding others constant.

        Parameters
        ----------
        param_name : str
            Parameter to vary.
        values : array-like
            Values to test.
        metric_fn : callable
            Function that takes a params dict and returns a scalar metric.

        Returns
        -------
        pd.DataFrame with columns: param_value, metric, delta_from_base.
        """
        base_metric = metric_fn(self._base)
        rows = []
        for v in values:
            params = dict(self._base)
            params[param_name] = v
            try:
                m = metric_fn(params)
                rows.append({
                    "param_value": v,
                    "metric": m,
                    "delta_from_base": m - base_metric,
                    "pct_change": (m - base_metric) / abs(base_metric) * 100 if base_metric != 0 else 0,
                })
            except Exception as exc:
                logger.warning("OAT failed for %s=%.4f: %s", param_name, v, exc)
                rows.append({
                    "param_value": v,
                    "metric": np.nan,
                    "delta_from_base": np.nan,
                    "pct_change": np.nan,
                })

        return pd.DataFrame(rows)

    def tornado_diagram(
        self,
        param_ranges: dict[str, tuple[float, float]],
        metric_fn: Callable[[dict[str, float]], float],
    ) -> pd.DataFrame:
        """
        Compute tornado diagram data: metric at low/high for each parameter.

        Parameters
        ----------
        param_ranges : dict
            {param_name: (low_value, high_value)}.
        metric_fn : callable
            Metric function.

        Returns
        -------
        pd.DataFrame sorted by impact (tornado ordering).
        Columns: param_name, base, low, high, impact.
        """
        base_metric = metric_fn(self._base)
        rows = []

        for name, (low, high) in param_ranges.items():
            # Low value
            params_low = dict(self._base)
            params_low[name] = low
            try:
                m_low = metric_fn(params_low)
            except Exception:
                m_low = np.nan

            # High value
            params_high = dict(self._base)
            params_high[name] = high
            try:
                m_high = metric_fn(params_high)
            except Exception:
                m_high = np.nan

            impact = abs(m_high - m_low) if not (np.isnan(m_high) or np.isnan(m_low)) else 0

            rows.append({
                "param_name": name,
                "base": base_metric,
                "low": m_low,
                "high": m_high,
                "impact": impact,
                "delta_low": m_low - base_metric,
                "delta_high": m_high - base_metric,
            })

        df = pd.DataFrame(rows).sort_values("impact", ascending=False)
        logger.info("Tornado diagram: %d parameters analyzed", len(df))
        return df

    def local_sensitivity(
        self,
        param_name: str,
        metric_fn: Callable[[dict[str, float]], float],
        epsilon_pct: float = 0.01,
    ) -> dict:
        """
        Compute local sensitivity (numerical derivative / elasticity).

        Elasticity = (df/f) / (dx/x) — percent change in output per percent change in input.

        Parameters
        ----------
        param_name : str
            Parameter to perturb.
        metric_fn : callable
            Metric function.
        epsilon_pct : float
            Relative perturbation size (default 1%).

        Returns
        -------
        dict with: derivative, elasticity, base_value, base_metric.
        """
        x0 = self._base[param_name]
        eps = abs(x0 * epsilon_pct)

        params_plus = dict(self._base)
        params_plus[param_name] = x0 + eps
        params_minus = dict(self._base)
        params_minus[param_name] = x0 - eps

        f_plus = metric_fn(params_plus)
        f_minus = metric_fn(params_minus)
        f0 = metric_fn(self._base)

        derivative = (f_plus - f_minus) / (2 * eps) if eps > 0 else 0
        elasticity = (derivative * x0 / f0) if f0 != 0 else 0

        return {
            "param_name": param_name,
            "derivative": float(derivative),
            "elasticity": float(elasticity),
            "base_value": x0,
            "base_metric": f0,
        }

    def full_sensitivity(
        self,
        param_names: list[str] | None = None,
        metric_fn: Callable[[dict[str, float]], float] | None = None,
        epsilon_pct: float = 0.01,
    ) -> pd.DataFrame:
        """
        Compute local elasticity for all (or selected) parameters.

        Returns DataFrame sorted by |elasticity| descending.
        """
        names = param_names or list(self._base.keys())
        if metric_fn is None:
            raise ValueError("metric_fn is required")

        rows = []
        for name in names:
            if name not in self._base:
                continue
            result = self.local_sensitivity(name, metric_fn, epsilon_pct)
            rows.append(result)

        df = pd.DataFrame(rows)
        if not df.empty:
            df["abs_elasticity"] = df["elasticity"].abs()
            df = df.sort_values("abs_elasticity", ascending=False)
        return df
