"""
Build context for agents - DO NOT IMPORT in production code.
Contains the ABC interfaces and types that all model/risk/selection modules must implement.
"""

# === ABC Interfaces (from src/core/base.py) ===
"""
class VolatilityModel(ABC):
    def fit(self, returns: pd.Series) -> Self
    def conditional_variance(self) -> pd.Series
    def forecast(self, steps: int = 10) -> tuple[np.ndarray, np.ndarray]
    def diagnose(self) -> DiagnosticsResult
    @property result -> VolatilityResult
    @property aic -> Optional[float]
    @property bic -> Optional[float]
    @property is_fitted -> bool

class SignalExtractor(ABC):
    def fit(self, spread: pd.Series) -> Self
    def get_signal_deviation(self) -> pd.Series
    @property result -> SignalResult

class RiskAnalyzer(ABC):
    def fit(self, returns: pd.Series, confidence: float = 0.99) -> Self
    def backtest(self, returns: pd.Series, var_series: pd.Series) -> BacktestResult
    @property result -> RiskResult
"""

# === Result Types (from src/core/types.py) ===
"""
@dataclass(frozen=True)
class DiagnosticsResult:
    ljung_box_stat: float
    ljung_box_pvalue: float
    arch_lm_stat: float
    arch_lm_pvalue: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    n_obs: int

@dataclass(frozen=True)
class VolatilityResult:
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

@dataclass(frozen=True)
class SignalResult:
    method: str
    signal: pd.Series
    trend: pd.Series
    deviation: pd.Series
    deviation_zscore: pd.Series
    signal_strength: float = 0.0
    is_overvalued: bool = False
    is_undervalued: bool = False

@dataclass(frozen=True)
class RiskResult:
    method: str
    confidence: float
    var: float
    es: float
    tail_index: Optional[float] = None
    threshold: Optional[float] = None
    gpd_shape: Optional[float] = None
    gpd_scale: Optional[float] = None
    n_exceedances: int = 0

@dataclass(frozen=True)
class RegimeResult:
    n_regimes: int
    labels: np.ndarray
    transition_matrix: np.ndarray
    regime_means: dict[int, float]
    regime_stds: dict[int, float]
    current_regime: int

@dataclass(frozen=True)
class BacktestResult:
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

@dataclass(frozen=True)
class ForecastTestResult:
    test_name: str
    model_a: str
    model_b: str
    dm_stat: Optional[float] = None
    dm_pvalue: Optional[float] = None
    winner: Optional[str] = None
    mcs_pvalue: Optional[float] = None
    in_confidence_set: bool = True
"""

# === Config (from src/core/config.py) ===
"""
class ModelConfig(BaseSettings):
    garch_p: int = 1
    garch_q: int = 1
    dist: str = "studentst"
    max_iter: int = 500
    ftol: float = 1e-4
    kalman_window: int = 60
    figarch_truncation: int = 500
    n_regimes: int = 3

class RiskConfig(BaseSettings):
    var_confidence: float = 0.99
    evt_threshold_percentile: float = 0.95
    backtest_window: int = 252
    es_confidence: float = 0.975
"""

# === Exceptions ===
"""
class NotFittedError(RuntimeError): pass
class ConvergenceError(RuntimeError): pass
class InsufficientDataError(ValueError): pass
"""
