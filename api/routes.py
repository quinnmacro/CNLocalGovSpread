"""
API route definitions.

All endpoints return JSON. Heavy computations are wrapped in
try/except with structured error responses.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.core.config import DataSource, get_settings
from src.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1")


# --- Response Models ---


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "4.0.0"
    data_source: str = ""


class DataSummary(BaseModel):
    n_rows: int
    n_columns: int
    date_range: list[str]
    columns: list[str]
    summary_stats: dict[str, dict[str, float]]


class ModelResult(BaseModel):
    model_name: str
    aic: float | None = None
    bic: float | None = None
    converged: bool = True
    params: dict[str, Any] = {}


class RiskMetrics(BaseModel):
    var_historical: float
    var_parametric: float
    var_evt: float
    es_evt: float
    gpd_shape: float | None = None
    gpd_scale: float | None = None


class ScenarioResponse(BaseModel):
    current_spread: float
    horizon: int
    n_paths: int
    median_final: float
    p5_final: float
    p95_final: float


# --- Data cache (in-memory for demo; production would use Redis/DB) ---

_data_cache: dict[str, Any] = {}


def _get_data() -> pd.DataFrame:
    """Load and cache data."""
    if "df" not in _data_cache:
        from src.core.data_engine import DataEngine
        engine = DataEngine()
        try:
            _data_cache["df"] = engine.load()
        except Exception:
            # Fallback to mock
            from src.core.config import DataConfig
            cfg = DataConfig(source=DataSource.MOCK)
            engine = DataEngine(cfg)
            _data_cache["df"] = engine.load()
    return _data_cache["df"]


def _get_returns() -> pd.Series:
    """Get cached returns series."""
    if "returns" not in _data_cache:
        df = _get_data()
        from src.core.data_engine import DataEngine
        engine = DataEngine()
        _data_cache["returns"] = engine.compute_returns(df)
    return _data_cache["returns"]


# --- Endpoints ---


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(data_source=settings.data.source.value)


@router.get("/data/summary", response_model=DataSummary)
async def data_summary() -> DataSummary:
    df = _get_data()
    spread_cols = [c for c in df.columns if c.startswith("spread_")]

    stats = {}
    for col in spread_cols:
        s = df[col]
        stats[col] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
        }

    return DataSummary(
        n_rows=len(df),
        n_columns=len(df.columns),
        date_range=[str(df["date"].iloc[0]), str(df["date"].iloc[-1])],
        columns=list(df.columns),
        summary_stats=stats,
    )


@router.get("/data/raw")
async def data_raw(
    limit: int = Query(default=100, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> dict:
    df = _get_data()
    sliced = df.iloc[offset:offset + limit]
    return {
        "data": json_safe_df(sliced),
        "total": len(df),
        "offset": offset,
        "limit": limit,
    }


@router.get("/models/fit")
async def fit_models(
    model_type: str = Query(default="garch", pattern="^(garch|egarch|gjr|ewma)$"),
) -> ModelResult:
    try:
        returns = _get_returns()

        if model_type == "ewma":
            from src.models.ewma import EWMAModel
            model = EWMAModel()
        else:
            from src.models.garch import GARCHModel
            model = GARCHModel(model_type=model_type)

        model.fit(returns)
        r = model.result

        return ModelResult(
            model_name=r.model_name,
            aic=r.aic,
            bic=r.bic,
            converged=r.converged,
            params={k: v for k, v in r.params.items() if isinstance(v, (int, float, str))},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/risk/metrics", response_model=RiskMetrics)
async def risk_metrics(
    confidence: float = Query(default=0.99, ge=0.9, le=0.999),
) -> RiskMetrics:
    returns = _get_returns()
    from src.risk.var_engine import VaREngine

    hist = VaREngine.historical_var(returns, confidence)
    param = VaREngine.parametric_var(returns, confidence)
    evt = VaREngine.evt_var(returns, confidence)

    return RiskMetrics(
        var_historical=hist["var"],
        var_parametric=param["var"],
        var_evt=evt["var"],
        es_evt=evt["es"],
        gpd_shape=evt.get("gpd_shape"),
        gpd_scale=evt.get("gpd_scale"),
    )


@router.get("/scenarios/generate", response_model=ScenarioResponse)
async def generate_scenarios(
    horizon: int = Query(default=252, ge=10, le=1000),
    n_paths: int = Query(default=5000, ge=100, le=50000),
) -> ScenarioResponse:
    df = _get_data()
    current = float(df["spread_all"].iloc[-1])
    returns = _get_returns()

    from src.analysis.scenarios import ScenarioGenerator
    gen = ScenarioGenerator.from_data(returns)
    result = gen.generate(current, horizon=horizon, n_paths=n_paths, seed=42)

    return ScenarioResponse(
        current_spread=current,
        horizon=horizon,
        n_paths=n_paths,
        median_final=float(result["median"].iloc[-1]),
        p5_final=float(result["p5"].iloc[-1]),
        p95_final=float(result["p95"].iloc[-1]),
    )


@router.get("/market/gauge")
async def market_gauge() -> dict:
    df = _get_data()
    returns = _get_returns()
    spread = df["spread_all"]

    from src.regime.market_gauge import MarketGauge
    gauge = MarketGauge()

    # Quick assessment (no vol model for speed)
    result = gauge.assess(spread=spread, returns=returns)

    return {
        "composite": result["composite"],
        "status": list(result["status"]),
        "indicators": {
            k: {"score": v["score"]} for k, v in result["indicator_scores"].items()
        },
    }


def json_safe_df(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to JSON-safe list of dicts."""
    records = df.to_dict(orient="records")
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, (pd.Timestamp, np.datetime64)):
                rec[k] = str(v)
            elif isinstance(v, (np.integer,)):
                rec[k] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[k] = float(v)
    return records
