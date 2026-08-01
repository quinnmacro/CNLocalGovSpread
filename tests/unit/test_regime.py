"""Unit tests for regime detection and market gauge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestHMMRegime:
    def test_fit(self, mock_returns):
        from src.models.ewma import EWMAModel
        from src.regime.hmm_regime import HMMRegimeDetector

        ewma = EWMAModel()
        ewma.fit(mock_returns)
        vol = ewma.result.volatility

        det = HMMRegimeDetector(n_regimes=3)
        result = det.fit(vol)
        assert result.n_regimes == 3
        assert len(result.labels) == len(vol)
        assert result.transition_matrix.shape == (3, 3)
        assert 0 <= result.current_regime < 3

    def test_transition_matrix_stochastic(self, mock_returns):
        """Each row of transition matrix should sum to ~1."""
        from src.models.ewma import EWMAModel
        from src.regime.hmm_regime import HMMRegimeDetector

        ewma = EWMAModel()
        ewma.fit(mock_returns)
        vol = ewma.result.volatility

        det = HMMRegimeDetector(n_regimes=3)
        result = det.fit(vol)
        row_sums = result.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


class TestMarketGauge:
    def test_assess(self, mock_data, mock_returns):
        from src.regime.market_gauge import MarketGauge
        gauge = MarketGauge()
        result = gauge.assess(spread=mock_data["spread_all"], returns=mock_returns)
        assert "composite" in result
        assert "status" in result
        assert "indicator_scores" in result
        assert 0 <= result["composite"] <= 100
        assert isinstance(result["status"], tuple)
        assert len(result["status"]) == 2

    def test_classify_boundaries(self):
        from src.regime.market_gauge import MarketGauge
        gauge = MarketGauge()
        eng_low, chn_low = gauge.classify(10)
        eng_high, chn_high = gauge.classify(90)
        assert eng_low != eng_high
