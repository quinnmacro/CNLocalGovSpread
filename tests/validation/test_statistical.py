"""
Statistical validation tests.

These tests verify that the statistical properties of the models
and simulations are correct, not just that they run without error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestSimulatorStatistics:
    """Verify Monte Carlo DGP produces correct distributional properties."""

    def test_mean_reversion(self):
        """AR(1) with phi<1 should be mean-reverting."""
        from src.core.simulator import SpreadSimulator
        sim = SpreadSimulator(phi=0.8, omega=0.02, alpha=0.08, beta=0.85, df_t=5.0)
        paths, _ = sim.simulate_multi_path(n_steps=1000, n_paths=200, seed=42)
        final_mean = np.mean(paths[:, -1])
        assert abs(final_mean) < 2.0

    def test_volatility_clustering(self):
        """GARCH should produce volatility clustering (autocorrelated squared returns)."""
        from src.core.simulator import SpreadSimulator
        sim = SpreadSimulator(phi=0.95, omega=0.02, alpha=0.15, beta=0.80, df_t=5.0)
        returns, _ = sim.simulate_single_path(n_steps=2000, seed=42)
        sq = returns ** 2
        autocorr = np.corrcoef(sq[:-1], sq[1:])[0, 1]
        assert autocorr > 0.05, f"GARCH should show vol clustering, got autocorr={autocorr:.4f}"

    def test_student_t_tails(self):
        """Student-t errors should produce fatter tails than normal."""
        from src.core.simulator import SpreadSimulator
        sim = SpreadSimulator(phi=0.0, omega=0.01, alpha=0.0, beta=0.0, df_t=4.0)
        returns, _ = sim.simulate_single_path(n_steps=5000, seed=42)
        from scipy.stats import kurtosis
        kurt = kurtosis(returns, fisher=False)
        assert kurt > 3.5, f"Student-t should have fat tails, got kurtosis={kurt:.2f}"


class TestVaRValidation:
    """Validate VaR estimates have correct coverage."""

    def test_historical_var_coverage(self, mock_returns):
        """VaR at 95% should be exceeded ~5% of the time."""
        from src.risk.var_engine import VaREngine
        r = VaREngine.historical_var(mock_returns, 0.95)
        var_val = r["var"]
        violations = (mock_returns > var_val).sum() / len(mock_returns)
        assert 0.01 < violations < 0.12, f"Expected ~5% violations, got {violations:.3f}"

    def test_var_monotonic_in_confidence(self, mock_returns):
        """Higher confidence should give higher VaR."""
        from src.risk.var_engine import VaREngine
        var_95 = VaREngine.historical_var(mock_returns, 0.95)["var"]
        var_99 = VaREngine.historical_var(mock_returns, 0.99)["var"]
        assert var_99 >= var_95, "VaR should increase with confidence level"

    def test_es_geq_var(self, mock_returns):
        """Expected Shortfall should always be >= VaR."""
        from src.risk.var_engine import VaREngine
        for conf in (0.95, 0.99, 0.995):
            r = VaREngine.historical_var(mock_returns, conf)
            assert r["es"] >= r["var"], f"ES should be >= VaR at {conf}"


class TestModelSelection:
    """Validate model selection identifies correct DGP."""

    def test_known_dgp_recovery(self):
        """When DGP is GARCH(1,1), GARCH should converge and fit well."""
        from src.core.simulator import SpreadSimulator
        from src.models.garch import GARCHModel
        from src.models.ewma import EWMAModel

        sim = SpreadSimulator(phi=0.0, omega=0.01, alpha=0.10, beta=0.85, df_t=6.0)
        returns, _ = sim.simulate_single_path(n_steps=2000, seed=123)
        returns = pd.Series(returns)

        garch = GARCHModel(model_type="garch")
        garch.fit(returns)
        assert garch.result.converged
        assert garch.result.aic is not None

        ewma = EWMAModel()
        ewma.fit(returns)

        # If EWMA also has AIC, GARCH should be competitive
        if ewma.result.aic is not None:
            assert garch.result.aic < ewma.result.aic + 50
