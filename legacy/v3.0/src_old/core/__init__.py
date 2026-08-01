"""
Core infrastructure for CNLocalGovSpread v4.0

This module provides:
- Configuration management (Pydantic models)
- Type definitions and protocols
- Abstract base classes for models
- Structured logging
"""

from src.core.config import Settings, DataConfig, RiskConfig, ModelConfig
from src.core.types import (
    VolatilityResult,
    SignalResult,
    RiskResult,
    RegimeResult,
    DiagnosticsResult,
    BacktestResult,
)
from src.core.base import VolatilityModel, SignalExtractor, RiskAnalyzer
from src.core.logging_config import get_logger, setup_logging

__all__ = [
    # Config
    "Settings",
    "DataConfig",
    "RiskConfig",
    "ModelConfig",
    # Types
    "VolatilityResult",
    "SignalResult",
    "RiskResult",
    "RegimeResult",
    "DiagnosticsResult",
    "BacktestResult",
    # Base classes
    "VolatilityModel",
    "SignalExtractor",
    "RiskAnalyzer",
    # Logging
    "get_logger",
    "setup_logging",
]
