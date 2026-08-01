"""
Monte Carlo spread simulator with AR(1) + GARCH(1,1) Student-t DGP.

Mathematical specification:
    shock[t]   = sigma[t] * z[t],          z[t] ~ iid Student-t(df)
    sigma2[t]  = omega + alpha * shock[t-1]^2 + beta * sigma2[t-1]
    spread[t]  = mu + phi * (spread[t-1] - mu) + shock[t]

This produces mean-reverting spreads with volatility clustering and fat tails,
matching the empirical stylised facts of China local-government bond spreads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class SimulatorParams:
    """Immutable parameter set for the AR(1)+GARCH DGP."""

    phi: float = 0.95          # AR(1) persistence
    omega: float = 0.02        # GARCH unconditional variance component
    alpha: float = 0.08        # GARCH ARCH coefficient
    beta: float = 0.88         # GARCH GARCH coefficient
    df_t: float = 5.0          # Student-t degrees of freedom
    mu: float = 0.0            # Long-run mean of spread changes

    def __post_init__(self) -> None:
        if self.alpha + self.beta >= 1.0:
            raise ValueError(
                f"GARCH non-stationary: alpha+beta={self.alpha + self.beta:.4f} >= 1"
            )
        if self.df_t <= 2.0:
            raise ValueError(f"Student-t df must be > 2 for finite variance, got {self.df_t}")

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def unconditional_variance(self) -> float:
        """Unconditional variance of the GARCH process."""
        return self.omega / (1.0 - self.persistence)


class SpreadSimulator:
    """
    Vectorised Monte Carlo simulator for spread paths.

    Supports single-path and multi-path simulation with reproducible seeding.
    """

    def __init__(
        self,
        phi: float = 0.95,
        omega: float = 0.02,
        alpha: float = 0.08,
        beta: float = 0.88,
        df_t: float = 5.0,
        mu: float = 0.0,
    ) -> None:
        self.params = SimulatorParams(
            phi=phi, omega=omega, alpha=alpha, beta=beta, df_t=df_t, mu=mu
        )

    @classmethod
    def from_params(cls, params: SimulatorParams) -> "SpreadSimulator":
        """Construct from an existing parameter set."""
        sim = cls.__new__(cls)
        sim.params = params
        return sim

    def simulate_single_path(
        self,
        n_steps: int = 1500,
        seed: int | None = 42,
        initial_spread: float = 0.0,
        initial_variance: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate one AR(1)+GARCH path.

        Returns:
            (spread_changes, conditional_volatility) arrays of shape (n_steps,).
        """
        rng = np.random.default_rng(seed)
        p = self.params

        if initial_variance is None:
            initial_variance = p.unconditional_variance

        # Pre-generate all Student-t innovations at once (vectorised)
        z = rng.standard_t(df=p.df_t, size=n_steps)

        shocks = np.empty(n_steps)
        sigma2 = np.empty(n_steps)
        spread = np.empty(n_steps)

        sigma2[0] = initial_variance
        shocks[0] = np.sqrt(sigma2[0]) * z[0]
        spread[0] = p.mu + p.phi * (initial_spread - p.mu) + shocks[0]

        for t in range(1, n_steps):
            sigma2[t] = p.omega + p.alpha * shocks[t - 1] ** 2 + p.beta * sigma2[t - 1]
            shocks[t] = np.sqrt(np.maximum(sigma2[t], 0.0)) * z[t]
            spread[t] = p.mu + p.phi * (spread[t - 1] - p.mu) + shocks[t]

        return spread, np.sqrt(np.maximum(sigma2, 0.0))

    def simulate_multi_path(
        self,
        n_steps: int = 500,
        n_paths: int = 1000,
        seed: int | None = None,
        initial_spread: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate multiple independent paths in parallel.

        Returns:
            (paths, vol_paths) arrays of shape (n_paths, n_steps).
        """
        rng = np.random.default_rng(seed)
        p = self.params
        unc_var = p.unconditional_variance

        # Generate all innovations at once: shape (n_paths, n_steps)
        z = rng.standard_t(df=p.df_t, size=(n_paths, n_steps))

        paths = np.empty((n_paths, n_steps))
        sigma2 = np.empty((n_paths, n_steps))
        shocks = np.empty((n_paths, n_steps))

        sigma2[:, 0] = unc_var
        shocks[:, 0] = np.sqrt(sigma2[:, 0]) * z[:, 0]
        paths[:, 0] = p.mu + p.phi * (initial_spread - p.mu) + shocks[:, 0]

        for t in range(1, n_steps):
            sigma2[:, t] = (
                p.omega + p.alpha * shocks[:, t - 1] ** 2 + p.beta * sigma2[:, t - 1]
            )
            shocks[:, t] = np.sqrt(np.maximum(sigma2[:, t], 0.0)) * z[:, t]
            paths[:, t] = p.mu + p.phi * (paths[:, t - 1] - p.mu) + shocks[:, t]

        return paths, np.sqrt(np.maximum(sigma2, 0.0))

    def simulate_scenario(
        self,
        current_spread: float,
        horizon: int = 252,
        n_paths: int = 5000,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Simulate forward scenarios from the current spread level.

        Returns DataFrame with columns: [path_0, ..., path_{n-1}, median, p5, p95].
        """
        paths, _ = self.simulate_multi_path(
            n_steps=horizon,
            n_paths=n_paths,
            seed=seed,
            initial_spread=current_spread,
        )

        # Convert changes to cumulative levels
        cumulative = current_spread + np.cumsum(paths, axis=1)

        dates = pd.bdate_range(start=pd.Timestamp.now(), periods=horizon, freq="B")
        result = pd.DataFrame(index=dates)

        # Summary statistics
        result["median"] = np.median(cumulative, axis=0)
        result["p5"] = np.percentile(cumulative, 5, axis=0)
        result["p25"] = np.percentile(cumulative, 25, axis=0)
        result["p75"] = np.percentile(cumulative, 75, axis=0)
        result["p95"] = np.percentile(cumulative, 95, axis=0)
        result["mean"] = np.mean(cumulative, axis=0)

        return result

    def __repr__(self) -> str:
        p = self.params
        return (
            f"SpreadSimulator(phi={p.phi}, omega={p.omega}, alpha={p.alpha}, "
            f"beta={p.beta}, df_t={p.df_t})"
        )
