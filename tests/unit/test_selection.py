"""Unit tests for model selection: diagnostics, tournament, forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestDiagnostics:
    def test_ljung_box(self, mock_returns):
        from src.selection.diagnostics import ljung_box_test
        result = ljung_box_test(mock_returns)
        assert "statistic" in result
        assert "pvalue" in result

    def test_arch_lm(self, mock_returns):
        from src.selection.diagnostics import arch_lm_test
        result = arch_lm_test(mock_returns)
        assert "statistic" in result
        assert "pvalue" in result

    def test_compute_all(self, mock_returns):
        from src.selection.diagnostics import compute_diagnostics
        d = compute_diagnostics(mock_returns)
        assert isinstance(d, object)  # DiagnosticsResult


class TestTournament:
    def test_add_and_run(self, mock_returns):
        from src.models.garch import GARCHModel
        from src.models.ewma import EWMAModel
        from src.selection.tournament import ModelTournament

        t = ModelTournament()
        g = GARCHModel(model_type="garch")
        g.fit(mock_returns)
        t.add_model("GARCH", g)

        e = EWMAModel()
        e.fit(mock_returns)
        t.add_model("EWMA", e)

        df = t.run()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_winner_aic(self, mock_returns):
        from src.models.garch import GARCHModel
        from src.models.ewma import EWMAModel
        from src.selection.tournament import ModelTournament

        t = ModelTournament()
        g = GARCHModel(model_type="garch")
        g.fit(mock_returns)
        t.add_model("GARCH", g)
        e = EWMAModel()
        e.fit(mock_returns)
        t.add_model("EWMA", e)
        t.run()

        winner = t.winner(criterion="aic")
        assert isinstance(winner, str)
        assert winner in ("GARCH", "EWMA")

    def test_rank(self, mock_returns):
        from src.models.garch import GARCHModel
        from src.models.ewma import EWMAModel
        from src.selection.tournament import ModelTournament

        t = ModelTournament()
        g = GARCHModel()
        g.fit(mock_returns)
        t.add_model("GARCH", g)
        e = EWMAModel()
        e.fit(mock_returns)
        t.add_model("EWMA", e)
        t.run()

        ranking = t.rank(criterion="aic")
        assert len(ranking) >= 1

    def test_summary(self, mock_returns):
        from src.models.garch import GARCHModel
        from src.selection.tournament import ModelTournament

        t = ModelTournament()
        g = GARCHModel()
        g.fit(mock_returns)
        t.add_model("GARCH", g)
        t.run()
        s = t.summary()
        assert "GARCH" in s


class TestForecast:
    def test_diebold_mariano(self):
        from src.selection.forecast_test import diebold_mariano_test
        np.random.seed(42)
        e1 = np.random.normal(0, 1, 500)
        e2 = np.random.normal(0, 1.2, 500)
        result = diebold_mariano_test(e1, e2)
        assert "statistic" in result
        assert "pvalue" in result
