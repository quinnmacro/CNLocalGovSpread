"""
Pydantic response models for all API endpoints.

Separates schema definitions from route logic for clarity
and enables OpenAPI auto-documentation.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# --- Data ---


class ColumnStatistics(BaseModel):
    """Per-column descriptive statistics with unit root test."""

    column: str
    n: int
    mean: float
    std: float
    skew: float
    kurtosis: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    adf_stat: Optional[float] = None
    adf_pvalue: Optional[float] = None
    is_stationary: Optional[bool] = None


class DataStatisticsResponse(BaseModel):
    """Response for GET /data/statistics."""

    columns: list[ColumnStatistics]


# --- Models ---


class TimePoint(BaseModel):
    """A single date-value pair for time series data."""

    date: str
    value: float


class DiagnosticsInfo(BaseModel):
    """Model diagnostic test results."""

    ljung_box_stat: float
    ljung_box_pvalue: float
    arch_lm_stat: float
    arch_lm_pvalue: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    has_arch_effects: bool
    is_normal: bool
    n_obs: int


class ModelDetailResponse(BaseModel):
    """Response for GET /models/{name}/detail and POST /models/fit-custom."""

    model_name: str
    params: dict[str, float]
    aic: Optional[float] = None
    bic: Optional[float] = None
    loglikelihood: Optional[float] = None
    converged: bool = True
    persistence: Optional[float] = None
    conditional_volatility: list[TimePoint] = Field(default_factory=list)
    standardized_residuals: list[TimePoint] = Field(default_factory=list)
    diagnostics: Optional[DiagnosticsInfo] = None


class TournamentRow(BaseModel):
    """A single row in the model tournament table."""

    model_name: str
    model_type: str
    aic: Optional[float] = None
    bic: Optional[float] = None
    converged: bool = True
    persistence: Optional[float] = None
    has_arch_effects: Optional[bool] = None
    is_normal: Optional[bool] = None
    lb_pvalue: Optional[float] = None
    arch_pvalue: Optional[float] = None
    jb_pvalue: Optional[float] = None


class TournamentResponse(BaseModel):
    """Response for GET /models/tournament."""

    models: list[TournamentRow]
    winner_aic: Optional[str] = None
    winner_bic: Optional[str] = None


class FigarchResponse(BaseModel):
    """Response for GET /models/figarch."""

    model_name: str
    d: float = Field(description="Fractional differencing parameter")
    params: dict[str, float]
    aic: Optional[float] = None
    bic: Optional[float] = None
    converged: bool = True
    persistence: Optional[float] = None
    conditional_volatility: list[TimePoint] = Field(default_factory=list)
    diagnostics: Optional[DiagnosticsInfo] = None


# --- Risk ---


class HillEstimate(BaseModel):
    """Hill tail index estimate at a given k percentile."""

    k_percentile: float
    tail_index: float
    shape: float


class HillInfo(BaseModel):
    """Hill estimator results."""

    tail_index: float
    shape: float
    threshold: float
    k: int
    estimates: list[HillEstimate] = Field(default_factory=list)


class MeanExcessPoint(BaseModel):
    """A single point on the mean excess plot."""

    threshold: float
    mean_excess: float


class GpdParams(BaseModel):
    """Generalized Pareto Distribution parameters."""

    xi: float = Field(description="Shape parameter")
    sigma: float = Field(description="Scale parameter")


class EvtResponse(BaseModel):
    """Response for GET /risk/evt."""

    hill: HillInfo
    mean_excess: list[MeanExcessPoint]
    gpd_params: GpdParams


class KupiecTest(BaseModel):
    """Kupiec unconditional coverage test."""

    statistic: float
    pvalue: float
    expected_rate: float
    actual_rate: float


class ChristoffersenTest(BaseModel):
    """Christoffersen conditional coverage test."""

    statistic: float
    pvalue: float


class BacktestResponse(BaseModel):
    """Response for GET /risk/backtest."""

    violations: int
    n_observations: int
    expected_violations: float
    actual_coverage: float
    passes: bool
    kupiec: KupiecTest
    christoffersen: ChristoffersenTest
    var_series: list[TimePoint] = Field(default_factory=list)


# --- Regimes ---


class RegimeLabel(BaseModel):
    """A single date-regime label."""

    date: str
    regime: int


class RegimeStats(BaseModel):
    """Per-regime statistics."""

    regime: int
    mean: float
    std: float


class HmmResponse(BaseModel):
    """Response for GET /regimes/hmm."""

    n_regimes: int
    labels: list[RegimeLabel]
    transition_matrix: list[list[float]]
    regime_stats: list[RegimeStats]
    current_regime: int
    current_regime_name: str


# --- Scenarios ---


class StressScenario(BaseModel):
    """A single stress test scenario."""

    name: str
    vol_multiplier: float
    median_final: float
    p5: float
    p95: float
    prob_exceed: float


class StressRequest(BaseModel):
    """Request body for POST /scenarios/stress."""

    current: float
    shock_multipliers: list[float] = Field(default=[1.0, 1.5, 2.0, 3.0])
    horizon: int = Field(default=60, ge=10, le=500)
    n_paths: int = Field(default=5000, ge=100, le=50000)


class StressResponse(BaseModel):
    """Response for POST /scenarios/stress."""

    scenarios: list[StressScenario]


# --- Sensitivity ---


class SensitivityVariable(BaseModel):
    """A single variable in the tornado diagram."""

    name: str
    low: float
    high: float
    base: float
    sensitivity_pct: float


class SensitivityResponse(BaseModel):
    """Response for GET /analysis/sensitivity."""

    base_value: float
    variables: list[SensitivityVariable]


# --- Custom fit request ---


class CustomFitRequest(BaseModel):
    """Request body for POST /models/fit-custom."""

    model_type: str = Field(pattern="^(garch|egarch|gjr)$")
    p: int = Field(default=1, ge=1, le=5)
    q: int = Field(default=1, ge=1, le=5)
    dist: str = Field(default="studentst", pattern="^(normal|studentst|skewt)$")


# --- Kalman Signal ---


class KalmanSignalResponse(BaseModel):
    """Response for GET /regimes/kalman-signal."""

    signal: list[TimePoint] = Field(default_factory=list)
    deviation: list[TimePoint] = Field(default_factory=list)
    deviation_zscore: list[TimePoint] = Field(default_factory=list)
    signal_strength: float
    is_overvalued: bool
    is_undervalued: bool
    sigma2_eta: float = Field(description="State transition variance")
    sigma2_eps: float = Field(description="Observation noise variance")
    q_ratio: float = Field(description="Signal-to-noise ratio σ²_η / σ²_ε")


# --- Change Points ---


class ChangepointSegment(BaseModel):
    """A single segment between two changepoints."""

    start_idx: int
    end_idx: int
    start_date: str
    end_date: str
    mean: float
    std: float


class ChangepointResponse(BaseModel):
    """Response for GET /regimes/changepoints."""

    method: str = Field(description="Detection method (pelt or binseg)")
    breakpoints: list[int] = Field(default_factory=list, description="Breakpoint indices")
    breakpoint_dates: list[str] = Field(default_factory=list)
    segments: list[ChangepointSegment] = Field(default_factory=list)
    n_segments: int
