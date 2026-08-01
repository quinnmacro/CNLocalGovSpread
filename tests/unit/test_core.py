"""Unit tests for core modules: config, data_engine, simulator, types."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestConfig:
    def test_default_settings(self):
        from src.core.config import get_settings, Settings
        s = get_settings()
        assert isinstance(s, Settings)
        assert s.data.source.value in ("mock", "csv", "wind")

    def test_data_config_explicit(self):
        from src.core.config import DataConfig, DataSource
        c = DataConfig(source=DataSource.MOCK)
        assert c.source == DataSource.MOCK
        assert c.mad_threshold > 0


class TestDataEngine:
    def test_load_mock(self, mock_data):
        assert isinstance(mock_data, pd.DataFrame)
        assert len(mock_data) > 0
        for col in ("date", "spread_all", "spread_5y", "spread_10y", "spread_30y"):
            assert col in mock_data.columns

    def test_no_nans_after_clean(self, mock_data):
        spread_cols = [c for c in mock_data.columns if c.startswith("spread_")]
        assert not mock_data[spread_cols].isna().any().any()

    def test_compute_returns(self, mock_returns):
        assert isinstance(mock_returns, pd.Series)
        assert len(mock_returns) > 0
        assert abs(mock_returns.mean()) < abs(mock_returns.std()) * 2


class TestSimulator:
    def test_single_path(self):
        from src.core.simulator import SpreadSimulator
        sim = SpreadSimulator(phi=0.95, omega=0.02, alpha=0.08, beta=0.88, df_t=5.0)
        returns, vol = sim.simulate_single_path(n_steps=500, seed=42)
        assert len(returns) == 500
        assert len(vol) == 500
        assert not np.any(np.isnan(returns))

    def test_multi_path(self):
        from src.core.simulator import SpreadSimulator
        sim = SpreadSimulator()
        paths, vols = sim.simulate_multi_path(n_steps=100, n_paths=10, seed=42)
        assert paths.shape == (10, 100)
        assert vols.shape == (10, 100)

    def test_scenario(self):
        from src.core.simulator import SpreadSimulator
        sim = SpreadSimulator()
        df = sim.simulate_scenario(current_spread=30.0, horizon=50, n_paths=100, seed=42)
        assert "median" in df.columns
        assert "p5" in df.columns
        assert "p95" in df.columns
        assert len(df) == 50


class TestTypes:
    def test_volatility_result(self):
        from src.core.types import VolatilityResult
        vr = VolatilityResult(
            model_name="test",
            params={"alpha": 0.1, "beta": 0.8},
            conditional_volatility=pd.Series([1.0, 2.0, 3.0]),
            standardized_residuals=pd.Series([0.1, -0.2, 0.3]),
            aic=100.0, bic=105.0,
            converged=True,
        )
        assert vr.model_name == "test"
        assert vr.converged is True
        assert vr.persistence is not None
        summary = vr.summary()
        assert "test" in summary

    def test_frozen_dataclass(self):
        from src.core.types import VolatilityResult
        vr = VolatilityResult(
            model_name="test",
            params={},
            conditional_volatility=pd.Series([1.0]),
            standardized_residuals=pd.Series([0.0]),
            aic=None, bic=None,
            converged=True,
        )
        with pytest.raises(AttributeError):
            vr.model_name = "other"
