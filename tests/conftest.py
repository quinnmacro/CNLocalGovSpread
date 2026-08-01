"""
Shared pytest fixtures for CNLocalGovSpread v4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def mock_data() -> pd.DataFrame:
    """Generate deterministic mock spread data for testing."""
    from src.core.data_engine import DataEngine
    from src.core.config import DataConfig, DataSource
    engine = DataEngine(DataConfig(source=DataSource.MOCK))
    return engine.load()


@pytest.fixture(scope="session")
def mock_returns(mock_data: pd.DataFrame) -> pd.Series:
    """Compute returns from mock data."""
    from src.core.data_engine import DataEngine
    from src.core.config import DataConfig, DataSource
    engine = DataEngine(DataConfig(source=DataSource.MOCK))
    return engine.compute_returns(mock_data)


@pytest.fixture(scope="session")
def fitted_garch(mock_returns: pd.Series):
    """Return a fitted GARCH model."""
    from src.models.garch import GARCHModel
    m = GARCHModel(model_type="garch")
    m.fit(mock_returns)
    return m


@pytest.fixture(scope="session")
def fitted_ewma(mock_returns: pd.Series):
    """Return a fitted EWMA model."""
    from src.models.ewma import EWMAModel
    m = EWMAModel()
    m.fit(mock_returns)
    return m
