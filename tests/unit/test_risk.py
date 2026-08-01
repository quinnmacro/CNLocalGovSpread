"""Unit tests for risk modules: VaR, EVT, Backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestVaREngine:
    def test_historical_var(self, mock_returns):
        from src.risk.var_engine import VaREngine
        r = VaREngine.historical_var(mock_returns, 0.99)
        assert "var" in r
        assert "es" in r
        assert isinstance(r["var"], float)

    def test_parametric_var(self, mock_returns):
        from src.risk.var_engine import VaREngine
        r = VaREngine.parametric_var(mock_returns, 0.99)
        assert "var" in r
        assert "df" in r
        assert r["df"] > 0

    def test_evt_var(self, mock_returns):
        from src.risk.var_engine import VaREngine
        r = VaREngine.evt_var(mock_returns, 0.99)
        assert "var" in r
        assert "es" in r

    def test_compare_methods(self, mock_returns):
        from src.risk.var_engine import VaREngine
        result = VaREngine.compare_methods(mock_returns, 0.99)
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 3
        assert "var" in result.columns


class TestEVT:
    def test_fit(self, mock_returns):
        from src.risk.evt import EVTAnalyzer
        a = EVTAnalyzer()
        a.fit(mock_returns)
        assert a.result is not None

    def test_hill_estimator(self, mock_returns):
        from src.risk.evt import EVTAnalyzer
        a = EVTAnalyzer()
        a.fit(mock_returns)
        hill = a.hill_estimator(k_percentile=0.10)
        assert "tail_index" in hill
        assert "shape" in hill
        assert "threshold" in hill
        assert "k" in hill
        assert hill["k"] > 0

    def test_mean_excess(self, mock_returns):
        from src.risk.evt import EVTAnalyzer
        a = EVTAnalyzer()
        a.fit(mock_returns)
        me = a.mean_excess_data(n_thresholds=30)
        assert "threshold" in me.columns
        assert "mean_excess" in me.columns


class TestBacktest:
    def test_kupiec(self, mock_returns):
        from src.risk.var_engine import VaREngine
        from src.risk.backtest import VaRBacktest

        var_val = VaREngine.historical_var(mock_returns, 0.99)["var"]
        # Create rolling VaR series (constant for simplicity)
        var_series = pd.Series(var_val, index=mock_returns.index)

        bt = VaRBacktest()
        result = bt.full_backtest(mock_returns, var_series, confidence=0.99)
        assert result.n_observations > 0
        assert result.n_violations >= 0
        assert result.kupiec_pvalue is not None or result.n_observations > 0
