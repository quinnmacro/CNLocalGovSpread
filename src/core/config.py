"""
Centralized configuration using Pydantic v2 Settings.

All settings can be overridden via environment variables with prefix CLS_.
Example: CLS_DATA__SOURCE=wind overrides DataConfig.source
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSource(str, Enum):
    MOCK = "mock"
    CSV = "csv"
    WIND = "wind"


class DataConfig(BaseSettings):
    """Data source and cleaning configuration."""

    model_config = SettingsConfigDict(env_prefix="CLS_DATA__")

    source: DataSource = DataSource.CSV
    csv_path: Path = Path("data/local_gov_spread.csv")
    start_date: str = "2018-01-01"
    mad_threshold: float = Field(default=5.0, description="MAD-based outlier detection threshold")
    ewma_lambda: float = Field(default=0.94, ge=0.8, le=0.99)


class RiskConfig(BaseSettings):
    """Risk measurement parameters."""

    model_config = SettingsConfigDict(env_prefix="CLS_RISK__")

    var_confidence: float = Field(default=0.99, ge=0.9, le=0.999)
    evt_threshold_percentile: float = Field(default=0.95, ge=0.85, le=0.99)
    backtest_window: int = Field(default=252, ge=60, description="Rolling backtest window in days")
    es_confidence: float = Field(default=0.975, ge=0.9, le=0.999)


class ModelConfig(BaseSettings):
    """Model selection and fitting parameters."""

    model_config = SettingsConfigDict(env_prefix="CLS_MODEL__")

    garch_p: int = Field(default=1, ge=0, le=5)
    garch_q: int = Field(default=1, ge=0, le=5)
    dist: str = Field(default="studentst", description="Error distribution: normal, studentst, skewt")
    max_iter: int = Field(default=500, ge=100)
    ftol: float = Field(default=1e-4, description="Optimization convergence tolerance")
    kalman_window: int = Field(default=60, ge=20)
    figarch_truncation: int = Field(default=500, ge=50, description="FIGARCH pi-weight truncation lag")
    n_regimes: int = Field(default=3, ge=2, le=5)


class DashboardConfig(BaseSettings):
    """Dash dashboard settings."""

    model_config = SettingsConfigDict(env_prefix="CLS_DASH__")

    host: str = "0.0.0.0"
    port: int = 8050
    debug: bool = False
    title: str = "CN Local Gov Spread | QuinnMacro"


class Settings(BaseSettings):
    """Master settings composing all sub-configs."""

    model_config = SettingsConfigDict(env_prefix="CLS_")

    data: DataConfig = Field(default_factory=DataConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)

    @field_validator("project_root", mode="before")
    @classmethod
    def resolve_root(cls, v: Any) -> Path:
        return Path(v).resolve() if v else Path(__file__).resolve().parent.parent.parent


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings instance."""
    return Settings()
