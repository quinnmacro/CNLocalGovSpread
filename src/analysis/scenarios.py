"""
Monte Carlo scenario generation for stress testing and fan charts.

Uses AR(1) + GARCH DGP with Student-t errors to generate forward
spread paths under various stress conditions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.core.logging_config import get_logger
from src.core.simulator import SpreadSimulator

logger = get_logger(__name__)


class ScenarioGenerator:
    """
    Monte Carlo scenario generator for spread analysis.

    Wraps SpreadSimulator with scenario-specific utilities.
    """

    def __init__(
        self,
        phi: float = 0.95,
        omega: float = 0.02,
        alpha: float = 0.08,
        beta: float = 0.88,
        df_t: float = 5.0,
    ) -> None:
        self._sim = SpreadSimulator(
            phi=phi, omega=omega, alpha=alpha, beta=beta, df_t=df_t,
        )

    @classmethod
    def from_data(cls, returns: pd.Series) -> "ScenarioGenerator":
        """
        Estimate DGP parameters from historical return data.

        Uses AR(1) OLS for phi, GARCH(1,1) for vol parameters.
        """
        r = returns.dropna().values
        n = len(r)

        # AR(1): r[t] = phi * r[t-1] + eps
        from numpy.polynomial.polynomial import polyfit
        if n > 10:
            x, y = r[:-1], r[1:]
            phi_est = float(np.corrcoef(x, y)[0, 1]) * np.std(y) / np.std(x)
            phi_est = np.clip(phi_est, 0.5, 0.99)
        else:
            phi_est = 0.95

        # Simple variance estimates
        var_total = float(np.var(r))
        omega_est = max(var_total * 0.04, 0.001)
        alpha_est = 0.08
        beta_est = min(0.88, 1 - alpha_est - omega_est / var_total) if var_total > 0 else 0.88

        # Student-t df via MLE
        from scipy import stats as sp_stats
        try:
            df_est = float(sp_stats.t.fit(r)[0])
            df_est = max(3.0, min(df_est, 30.0))
        except Exception:
            df_est = 5.0

        return cls(phi=phi_est, omega=omega_est, alpha=alpha_est, beta=beta_est, df_t=df_est)

    def generate(
        self,
        current_spread: float,
        horizon: int = 252,
        n_paths: int = 5000,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate forward scenario paths.

        Returns DataFrame indexed by business dates with percentile bands.
        """
        return self._sim.simulate_scenario(
            current_spread=current_spread,
            horizon=horizon,
            n_paths=n_paths,
            seed=seed,
        )

    def stress_test(
        self,
        current_spread: float,
        shock_multipliers: list[float] | None = None,
        horizon: int = 60,
        n_paths: int = 5000,
        threshold_bps: float = 10.0,
    ) -> dict:
        """
        Stress test with elevated volatility.

        Parameters
        ----------
        shock_multipliers : list[float]
            Volatility multipliers (e.g., [1.0, 2.0, 3.0]).
        threshold_bps : float
            Threshold for probability computation (spread widening in bps).

        Returns
        -------
        dict : {scenario_name: {probability, median, p95, mean}}
        """
        if shock_multipliers is None:
            shock_multipliers = [1.0, 1.5, 2.0, 3.0]

        results = {}
        base_omega = self._sim.params.omega

        for mult in shock_multipliers:
            # Increase omega to produce higher volatility
            stress_omega = base_omega * mult ** 2
            stress_sim = SpreadSimulator(
                phi=self._sim.params.phi,
                omega=stress_omega,
                alpha=self._sim.params.alpha,
                beta=self._sim.params.beta,
                df_t=self._sim.params.df_t,
            )

            paths, _ = stress_sim.simulate_multi_path(
                n_steps=horizon, n_paths=n_paths, seed=42,
                initial_spread=0.0,
            )
            cumulative = current_spread + np.cumsum(paths, axis=1)
            final = cumulative[:, -1]

            prob_exceed = float(np.mean(final > current_spread + threshold_bps))
            prob_decline = float(np.mean(final < current_spread - threshold_bps))

            name = f"vol_x{mult:.1f}"
            results[name] = {
                "probability_exceed_threshold": prob_exceed,
                "probability_decline_threshold": prob_decline,
                "median_final": float(np.median(final)),
                "p5_final": float(np.percentile(final, 5)),
                "p95_final": float(np.percentile(final, 95)),
                "mean_final": float(np.mean(final)),
                "vol_multiplier": mult,
            }

        logger.info("Stress test complete: %d scenarios", len(results))
        return results

    def fan_chart_data(
        self,
        current_spread: float,
        horizon: int = 252,
        n_paths: int = 5000,
        seed: int | None = 42,
    ) -> dict:
        """
        Generate fan chart data for Plotly visualization.

        Returns dict with percentile bands suitable for area plots.
        """
        df = self.generate(current_spread, horizon, n_paths, seed)
        return {
            "dates": df.index.tolist(),
            "median": df["median"].tolist(),
            "p5": df["p5"].tolist(),
            "p25": df["p25"].tolist(),
            "p75": df["p75"].tolist(),
            "p95": df["p95"].tolist(),
            "mean": df["mean"].tolist(),
            "current": current_spread,
        }
