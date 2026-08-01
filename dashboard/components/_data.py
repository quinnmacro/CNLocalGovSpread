"""
Shared data cache for dashboard pages.

All heavy computations happen once and are cached in-memory.
Each page calls the getter it needs; the first call triggers computation.
"""

from __future__ import annotations

import pandas as pd

_CACHE: dict[str, object] = {}


def get_engine():
    from src.core.data_engine import DataEngine
    if "engine" not in _CACHE:
        _CACHE["engine"] = DataEngine()
    return _CACHE["engine"]


def get_data() -> pd.DataFrame:
    if "df" not in _CACHE:
        _CACHE["df"] = get_engine().load()
    return _CACHE["df"]  # type: ignore[return-value]


def get_returns(column: str = "spread_all") -> pd.Series:
    key = f"returns_{column}"
    if key not in _CACHE:
        _CACHE[key] = get_engine().compute_returns(get_data(), column)
    return _CACHE[key]  # type: ignore[return-value]


def get_fitted_models() -> dict:
    """Fit GARCH, EGARCH, GJR, EWMA and return {name: model}."""
    if "fitted_models" not in _CACHE:
        returns = get_returns()
        from src.models.garch import GARCHModel
        from src.models.ewma import EWMAModel

        models = {}
        for mtype in ("garch", "egarch", "gjr"):
            try:
                m = GARCHModel(model_type=mtype)
                m.fit(returns)
                models[mtype.upper()] = m
            except Exception:
                pass

        try:
            ewma = EWMAModel()
            ewma.fit(returns)
            models["EWMA"] = ewma
        except Exception:
            pass

        _CACHE["fitted_models"] = models
    return _CACHE["fitted_models"]  # type: ignore[return-value]


def get_tournament_df() -> pd.DataFrame:
    if "tournament_df" not in _CACHE:
        from src.selection.tournament import ModelTournament
        t = ModelTournament()
        for name, model in get_fitted_models().items():
            t.add_model(name, model)
        _CACHE["tournament_df"] = t.run()
    return _CACHE["tournament_df"]  # type: ignore[return-value]


def get_var_results(confidence: float = 0.99) -> dict[str, dict]:
    if "var_results" not in _CACHE:
        from src.risk.var_engine import VaREngine
        returns = get_returns()
        _CACHE["var_results"] = {
            "Historical": VaREngine.historical_var(returns, confidence),
            "Parametric-t": VaREngine.parametric_var(returns, confidence),
            "EVT-GPD": VaREngine.evt_var(returns, confidence),
        }
    return _CACHE["var_results"]  # type: ignore[return-value]


def get_evt_analyzer():
    if "evt" not in _CACHE:
        from src.risk.evt import EVTAnalyzer
        a = EVTAnalyzer()
        a.fit(get_returns())
        _CACHE["evt"] = a
    return _CACHE["evt"]


def get_regime_result():
    if "regime" not in _CACHE:
        from src.regime.hmm_regime import HMMRegimeDetector
        from src.models.ewma import EWMAModel
        returns = get_returns()
        ewma = EWMAModel()
        ewma.fit(returns)
        vol = ewma.result.volatility
        det = HMMRegimeDetector(n_regimes=3)
        _CACHE["regime"] = (det.fit(vol), vol)
    return _CACHE["regime"]


def get_market_gauge() -> dict:
    if "gauge" not in _CACHE:
        from src.regime.market_gauge import MarketGauge
        df = get_data()
        returns = get_returns()
        gauge = MarketGauge()
        _CACHE["gauge"] = gauge.assess(spread=df["spread_all"], returns=returns)
    return _CACHE["gauge"]  # type: ignore[return-value]


def get_scenario_data(horizon: int = 252, n_paths: int = 5000):
    if "scenario" not in _CACHE:
        from src.analysis.scenarios import ScenarioGenerator
        returns = get_returns()
        gen = ScenarioGenerator.from_data(returns)
        df = get_data()
        current = float(df["spread_all"].iloc[-1])
        fan = gen.fan_chart_data(current, horizon=horizon, n_paths=n_paths, seed=42)
        stress = gen.stress_test(
            current,
            shock_multipliers=[1.0, 1.5, 2.0, 3.0],
            horizon=60,
            n_paths=n_paths,
        )
        _CACHE["scenario"] = {"fan": fan, "stress": stress, "current": current}
    return _CACHE["scenario"]
