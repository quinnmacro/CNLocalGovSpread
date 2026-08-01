"""
Integration test: full pipeline from data → models → risk → selection → regime.
"""

from __future__ import annotations

import pandas as pd
import pytest


class TestFullPipeline:
    """End-to-end test of the complete analysis pipeline."""

    def test_data_to_risk(self, mock_data, mock_returns):
        """Data → fit models → compute risk → tournament → regime."""

        # 1. Fit volatility models
        from src.models.garch import GARCHModel
        from src.models.ewma import EWMAModel

        garch = GARCHModel(model_type="garch")
        garch.fit(mock_returns)
        assert garch.result is not None
        assert garch.result.converged

        ewma = EWMAModel()
        ewma.fit(mock_returns)
        assert ewma.result is not None

        # 2. Model tournament
        from src.selection.tournament import ModelTournament
        tournament = ModelTournament()
        tournament.add_model("GARCH", garch)
        tournament.add_model("EWMA", ewma)
        df = tournament.run()
        assert len(df) == 2
        winner = tournament.winner("aic")
        assert winner in ("GARCH", "EWMA")

        # 3. Risk metrics
        from src.risk.var_engine import VaREngine
        var_hist = VaREngine.historical_var(mock_returns, 0.99)
        var_param = VaREngine.parametric_var(mock_returns, 0.99)
        var_evt = VaREngine.evt_var(mock_returns, 0.99)
        assert var_hist["var"] != 0
        assert var_param["var"] != 0
        assert var_evt["var"] != 0

        # 4. EVT diagnostics
        from src.risk.evt import EVTAnalyzer
        evt = EVTAnalyzer()
        evt.fit(mock_returns)
        hill = evt.hill_estimator()
        assert "tail_index" in hill
        assert "shape" in hill

        # 5. Regime detection
        from src.regime.hmm_regime import HMMRegimeDetector
        vol = ewma.result.conditional_volatility
        det = HMMRegimeDetector(n_regimes=3)
        regime = det.fit(vol)
        assert regime.n_regimes == 3

        # 6. Market gauge
        from src.regime.market_gauge import MarketGauge
        gauge = MarketGauge()
        assess = gauge.assess(spread=mock_data["spread_all"], returns=mock_returns)
        assert 0 <= assess["composite"] <= 100

        # 7. Scenario generation
        from src.analysis.scenarios import ScenarioGenerator
        gen = ScenarioGenerator.from_data(mock_returns)
        fan = gen.fan_chart_data(
            current_spread=float(mock_data["spread_all"].iloc[-1]),
            horizon=30, n_paths=500, seed=42,
        )
        assert "median" in fan
        assert len(fan["median"]) == 30

    def test_reporting(self, mock_returns, fitted_garch, fitted_ewma, tmp_path):
        """Test report generation."""
        from src.selection.tournament import ModelTournament
        from src.reporting.report import ReportGenerator

        t = ModelTournament()
        t.add_model("GARCH", fitted_garch)
        t.add_model("EWMA", fitted_ewma)
        t_df = t.run()

        rg = ReportGenerator()
        rg.add_section("Summary", "Pipeline test report")
        rg.add_summary_table("Tournament", t_df)

        # HTML
        html_path = tmp_path / "report.html"
        html_str = rg.generate_html(output_path=html_path)
        assert html_path.exists()
        assert len(html_str) > 100

        # JSON
        json_str = rg.generate_json()
        assert len(json_str) > 50
