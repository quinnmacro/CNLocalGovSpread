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
        assert r["var"] <= r["es"]  # VaR <= ES in upper tail
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
        assert isinstance(result, dict)
        assert len(result) >= 3


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
        assert "k_values" in hill
        assert "xi_values" in hill
        assert len(hill["k_values"]) > 0

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
        # Create rolling VaR series
        var_series = pd.Series(var_val, index=mock_returns.index)

        bt = VaRBacktest()
        result = bt.backtest(mock_returns, var_series, confidence=0.99)
        assert "kupiec_pvalue" in result
        assert "n_violations" in result
        assert result["n_violations"] >= 0
