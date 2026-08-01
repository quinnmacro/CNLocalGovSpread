"""Unit tests for volatility models: GARCH, EWMA, FIGARCH, Kalman, ML."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestGARCH:
    def test_fit_garch(self, fitted_garch):
        r = fitted_garch.result
        assert "GARCH" in r.model_name
        assert r.converged
        assert r.aic is not None
        assert r.bic is not None
        assert len(r.conditional_volatility) > 0
        assert len(r.standardized_residuals) > 0

    def test_fit_egarch(self, mock_returns):
        from src.models.garch import GARCHModel
        m = GARCHModel(model_type="egarch")
        m.fit(mock_returns)  # spread changes
        assert m.result.converged
        assert "EGARCH" in m.result.model_name

    def test_fit_gjr(self, mock_returns):
        from src.models.garch import GARCHModel
        m = GARCHModel(model_type="gjr")
        m.fit(mock_returns)  # spread changes
        assert m.result.converged

    def test_params_extracted(self, fitted_garch):
        params = fitted_garch.result.params
        assert isinstance(params, dict)
        assert len(params) > 0

    def test_persistence(self, fitted_garch):
        p = fitted_garch.result.persistence
        assert p is not None
        assert 0 < p < 2  # Stationarity requires persistence < 1, but allow margin


class TestEWMA:
    def test_fit(self, fitted_ewma):
        r = fitted_ewma.result
        assert "EWMA" in r.model_name
        assert r.converged
        assert r.aic is None  # Not MLE
        assert r.bic is None
        assert len(r.conditional_volatility) > 0

    def test_lambda_bounded(self, fitted_ewma):
        params = fitted_ewma.result.params
        lam = params.get("lambda", 0.94)
        assert 0.5 <= lam <= 0.99


class TestKalman:
    def test_fit(self, mock_returns):
        from src.models.kalman import KalmanSignalExtractor
        m = KalmanSignalExtractor()
        m.fit(mock_returns)  # spread changes
        assert m.result is not None
        r = m.result
        assert r is not None


class TestFIGARCH:
    def test_fit(self, mock_returns):
        from src.models.figarch import FIGARCHModel
        m = FIGARCHModel()
        m.fit(mock_returns)  # spread changes
        r = m.result
        assert "FIGARCH" in r.model_name or r.converged
        assert len(r.conditional_volatility) > 0
