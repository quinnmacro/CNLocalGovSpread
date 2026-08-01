"""
Immutable result types for all analytical outputs.

Every analysis function returns one of these frozen dataclasses.
This ensures reproducibility and makes results serializable/testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DiagnosticsResult:
    """Statistical diagnostics for a fitted model."""

    ljung_box_stat: float
    ljung_box_pvalue: float
    arch_lm_stat: float
    arch_lm_pvalue: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    n_obs: int

    @property
    def has_arch_effects(self) -> bool:
        return self.arch_lm_pvalue < 0.05

    @property
    def is_normal(self) -> bool:
        return self.jarque_bera_pvalue > 0.05

    def summary(self) -> str:
        lines = [
            f"Diagnostics (n={self.n_obs}):",
            f"  Ljung-Box:  Q={self.ljung_box_stat:.3f}, p={self.ljung_box_pvalue:.4f}",
            f"  ARCH-LM:    F={self.arch_lm_stat:.3f}, p={self.arch_lm_pvalue:.4f}",
            f"  Jarque-Bera: JB={self.jarque_bera_stat:.3f}, p={self.jarque_bera_pvalue:.4f}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class VolatilityResult:
    """Result from a fitted volatility model (GARCH/EGARCH/FIGARCH/EWMA)."""

    model_name: str
    params: dict[str, float]
    conditional_volatility: pd.Series
    standardized_residuals: pd.Series
    aic: Optional[float] = None
    bic: Optional[float] = None
    loglikelihood: Optional[float] = None
    converged: bool = True
    diagnostics: Optional[DiagnosticsResult] = None
    forecast_mean: Optional[np.ndarray] = None
    forecast_vol: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if len(self.conditional_volatility) == 0:
            raise ValueError("conditional_volatility must not be empty")

    @property
    def persistence(self) -> Optional[float]:
        """GARCH persistence (alpha + beta). None for non-GARCH models."""
        alpha = self.params.get("alpha", self.params.get("alpha[1]"))
        beta = self.params.get("beta", self.params.get("beta[1]"))
        if alpha is not None and beta is not None:
            return float(alpha) + float(beta)
        return None

    def summary(self) -> str:
        lines = [
            f"VolatilityResult: {self.model_name}",
            f"  Converged: {self.converged}",
            f"  AIC={self.aic:.2f}, BIC={self.bic:.2f}" if self.aic else "  AIC/BIC: N/A",
            f"  Params: {self.params}",
        ]
        if self.persistence is not None:
            lines.append(f"  Persistence (α+β): {self.persistence:.4f}")
        return "\n".join(lines)


@dataclass(frozen=True)
class SignalResult:
    """Signal extraction result from Kalman filter or trend decomposition."""

    method: str
    signal: pd.Series
    trend: pd.Series
    deviation: pd.Series
    deviation_zscore: pd.Series
    signal_strength: float = field(default=0.0)
    is_overvalued: bool = False
    is_undervalued: bool = False

    @property
    def current_deviation(self) -> float:
        return float(self.deviation.iloc[-1])

    def summary(self) -> str:
        return (
            f"SignalResult ({self.method}):\n"
            f"  Current signal: {self.signal.iloc[-1]:.2f}\n"
            f"  Deviation: {self.current_deviation:.2f} (z={self.deviation_zscore.iloc[-1]:.2f})\n"
            f"  Signal strength: {self.signal_strength:.2f}"
        )


@dataclass(frozen=True)
class RiskResult:
    """Value-at-Risk and Expected Shortfall result."""

    method: str  # "historical", "parametric", "evt"
    confidence: float
    var: float
    es: float
    tail_index: Optional[float] = None
    threshold: Optional[float] = None
    gpd_shape: Optional[float] = None
    gpd_scale: Optional[float] = None
    n_exceedances: int = 0

    @property
    def is_heavy_tailed(self) -> bool:
        return self.gpd_shape is not None and self.gpd_shape > 0

    def summary(self) -> str:
        lines = [
            f"RiskResult ({self.method}, {self.confidence:.1%}):",
            f"  VaR = {self.var:.4f}",
            f"  ES  = {self.es:.4f}",
        ]
        if self.gpd_shape is not None:
            lines.append(f"  GPD: ξ={self.gpd_shape:.4f}, σ={self.gpd_scale:.4f}")
        return "\n".join(lines)


@dataclass(frozen=True)
class RegimeResult:
    """Hidden Markov Model regime detection result."""

    n_regimes: int
    labels: np.ndarray
    transition_matrix: np.ndarray
    regime_means: dict[int, float]
    regime_stds: dict[int, float]
    current_regime: int

    @property
    def current_regime_name(self) -> str:
        names = {0: "Low Vol", 1: "Mid Vol", 2: "High Vol", 3: "Extreme"}
        return names.get(self.current_regime, f"Regime-{self.current_regime}")

    @property
    def stationary_distribution(self) -> np.ndarray:
        """Compute stationary distribution from transition matrix."""
        n = self.transition_matrix.shape[0]
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        return stationary / stationary.sum()

    def summary(self) -> str:
        lines = [f"RegimeResult (n_regimes={self.n_regimes}):"]
        lines.append(f"  Current regime: {self.current_regime_name}")
        for i in range(self.n_regimes):
            lines.append(
                f"  Regime {i}: μ={self.regime_means[i]:.2f}, σ={self.regime_stds[i]:.2f}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class BacktestResult:
    """VaR/ES backtesting result with coverage tests."""

    method: str
    n_observations: int
    n_violations: int
    expected_violations: float
    actual_coverage: float
    kupiec_stat: Optional[float] = None
    kupiec_pvalue: Optional[float] = None
    christoffersen_stat: Optional[float] = None
    christoffersen_pvalue: Optional[float] = None
    passes: bool = True

    @property
    def violation_ratio(self) -> float:
        return self.n_violations / self.expected_violations if self.expected_violations > 0 else float("inf")

    def summary(self) -> str:
        lines = [
            f"BacktestResult ({self.method}):",
            f"  Violations: {self.n_violations}/{self.n_observations} "
            f"(expected {self.expected_violations:.1f})",
            f"  Coverage: {self.actual_coverage:.4f}",
            f"  Passes: {self.passes}",
        ]
        if self.kupiec_pvalue is not None:
            lines.append(f"  Kupiec: stat={self.kupiec_stat:.3f}, p={self.kupiec_pvalue:.4f}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ForecastTestResult:
    """Diebold-Mariano test or Model Confidence Set result."""

    test_name: str
    model_a: str
    model_b: str
    dm_stat: Optional[float] = None
    dm_pvalue: Optional[float] = None
    winner: Optional[str] = None
    mcs_pvalue: Optional[float] = None
    in_confidence_set: bool = True

    def summary(self) -> str:
        lines = [f"ForecastTestResult ({self.test_name}):"]
        lines.append(f"  {self.model_a} vs {self.model_b}")
        if self.dm_stat is not None:
            lines.append(f"  DM stat={self.dm_stat:.3f}, p={self.dm_pvalue:.4f}")
            lines.append(f"  Winner: {self.winner}")
        if self.mcs_pvalue is not None:
            lines.append(f"  MCS p={self.mcs_pvalue:.4f}, in_set={self.in_confidence_set}")
        return "\n".join(lines)
