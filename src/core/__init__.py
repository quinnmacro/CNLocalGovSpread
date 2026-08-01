"""
Core module: configuration, types, abstract interfaces, data engine, simulator.
"""

from src.core.config import (
    DashboardConfig,
    DataConfig,
    DataSource,
    ModelConfig,
    RiskConfig,
    Settings,
    get_settings,
)
from src.core.types import (
    BacktestResult,
    DiagnosticsResult,
    ForecastTestResult,
    RegimeResult,
    RiskResult,
    SignalResult,
    VolatilityResult,
)
from src.core.base import (
    ConvergenceError,
    InsufficientDataError,
    NotFittedError,
    RiskAnalyzer,
    SignalExtractor,
    VolatilityModel,
)
from src.core.data_engine import DataEngine
from src.core.wind_client import WindClient, DEFAULT_SPREAD_CODES, CREDIT_SPREAD_CODES
from src.core.simulator import SimulatorParams, SpreadSimulator
from src.core.logging_config import get_logger, setup_logging

__all__ = [
    # Config
    "Settings", "get_settings", "DataConfig", "DataSource",
    "RiskConfig", "ModelConfig", "DashboardConfig",
    # Types
    "DiagnosticsResult", "VolatilityResult", "SignalResult",
    "RiskResult", "RegimeResult", "BacktestResult", "ForecastTestResult",
    # Base
    "VolatilityModel", "SignalExtractor", "RiskAnalyzer",
    "NotFittedError", "ConvergenceError", "InsufficientDataError",
    # Data
    "DataEngine", "WindClient", "DEFAULT_SPREAD_CODES", "CREDIT_SPREAD_CODES", "SpreadSimulator", "SimulatorParams",
    # Logging
    "get_logger", "setup_logging",
]
