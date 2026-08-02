"""
API route definitions.

All endpoints return JSON. Heavy computations are wrapped in
try/except with structured error responses.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel

from api.schemas import (
    BayesianSTSResponse,
    STSSignalResponse,
    BacktestResponse,
    ChangepointResponse,
    ChangepointSegment,
    ChristoffersenTest,
    ColumnStatistics,
    CustomFitRequest,
    DataStatisticsResponse,
    DiagnosticsInfo,
    EvtResponse,
    FigarchResponse,
    GpdParams,
    HillEstimate,
    HillInfo,
    HmmResponse,
    KalmanSignalResponse,
    KupiecTest,
    MeanExcessPoint,
    ModelDetailResponse,
    RegimeLabel,
    RegimeStats,
    SensitivityResponse,
    SensitivityVariable,
    StressRequest,
    StressResponse,
    StressScenario,
    TimePoint,
    TournamentResponse,
    TournamentRow,
)
from src.core.config import DataSource, get_settings
from src.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1")


# --- Legacy Response Models (kept for existing endpoints) ---


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


# --- Helper functions ---


def _safe_float(v: Any) -> float:
    """Convert any numeric type to a JSON-safe Python float."""
    if v is None:
        return 0.0
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return 0.0
    return float(v)


def _series_to_timepoints(series: pd.Series) -> list[TimePoint]:
    """Convert a pandas Series with DatetimeIndex to list of TimePoint."""
    points = []
    for idx, val in series.items():
        date_str = str(idx.date()) if hasattr(idx, "date") else str(idx)
        points.append(TimePoint(date=date_str, value=_safe_float(val)))
    return points


def _build_diagnostics_info(model: Any) -> DiagnosticsInfo | None:
    """Run diagnostics on a fitted model and return DiagnosticsInfo."""
    try:
        diag = model.diagnose()
        return DiagnosticsInfo(
            ljung_box_stat=_safe_float(diag.ljung_box_stat),
            ljung_box_pvalue=_safe_float(diag.ljung_box_pvalue),
            arch_lm_stat=_safe_float(diag.arch_lm_stat),
            arch_lm_pvalue=_safe_float(diag.arch_lm_pvalue),
            jarque_bera_stat=_safe_float(diag.jarque_bera_stat),
            jarque_bera_pvalue=_safe_float(diag.jarque_bera_pvalue),
            has_arch_effects=bool(diag.has_arch_effects),
            is_normal=bool(diag.is_normal),
            n_obs=int(diag.n_obs),
        )
    except Exception:
        return None


def _build_model_detail(model: Any) -> ModelDetailResponse:
    """Build a ModelDetailResponse from a fitted VolatilityModel."""
    r = model.result
    diag_info = _build_diagnostics_info(model)

    return ModelDetailResponse(
        model_name=r.model_name,
        params={k: _safe_float(v) for k, v in r.params.items()},
        aic=_safe_float(r.aic) if r.aic is not None else None,
        bic=_safe_float(r.bic) if r.bic is not None else None,
        loglikelihood=_safe_float(r.loglikelihood) if r.loglikelihood is not None else None,
        converged=bool(r.converged),
        persistence=_safe_float(r.persistence) if r.persistence is not None else None,
        conditional_volatility=_series_to_timepoints(r.conditional_volatility),
        standardized_residuals=_series_to_timepoints(r.standardized_residuals),
        diagnostics=diag_info,
    )


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


# =====================================================================
# Existing endpoints (unchanged)
# =====================================================================


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


# =====================================================================
# New endpoints
# =====================================================================


# ---- 1. Data statistics with ADF test ----


@router.get("/data/statistics", response_model=DataStatisticsResponse)
async def data_statistics() -> DataStatisticsResponse:
    """Column statistics with Augmented Dickey-Fuller unit root test."""
    try:
        from statsmodels.tsa.stattools import adfuller

        df = _get_data()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        columns_stats: list[ColumnStatistics] = []

        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) < 10:
                continue

            n = len(s)
            mean_val = _safe_float(s.mean())
            std_val = _safe_float(s.std())
            skew_val = _safe_float(s.skew())
            kurt_val = _safe_float(s.kurtosis())
            min_val = _safe_float(s.min())
            max_val = _safe_float(s.max())
            median_val = _safe_float(s.median())
            q25_val = _safe_float(s.quantile(0.25))
            q75_val = _safe_float(s.quantile(0.75))

            # ADF test
            adf_stat: float | None = None
            adf_pval: float | None = None
            is_stat: bool | None = None
            try:
                adf_result = adfuller(s.values, autolag="AIC")
                adf_stat = _safe_float(adf_result[0])
                adf_pval = _safe_float(adf_result[1])
                is_stat = bool(adf_pval < 0.05)
            except Exception:
                pass

            columns_stats.append(ColumnStatistics(
                column=col,
                n=n,
                mean=mean_val,
                std=std_val,
                skew=skew_val,
                kurtosis=kurt_val,
                min=min_val,
                max=max_val,
                median=median_val,
                q25=q25_val,
                q75=q75_val,
                adf_stat=adf_stat,
                adf_pvalue=adf_pval,
                is_stationary=is_stat,
            ))

        return DataStatisticsResponse(columns=columns_stats)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---- 2. Model tournament ----


@router.get("/models/tournament", response_model=TournamentResponse)
async def model_tournament() -> TournamentResponse:
    """Full model comparison: GARCH, EGARCH, GJR, EWMA, FIGARCH."""
    try:
        returns = _get_returns()

        from src.models.ewma import EWMAModel
        from src.models.figarch import FIGARCHModel
        from src.models.garch import GARCHModel
        from src.selection.tournament import ModelTournament

        # Fit all models
        models_to_fit: dict[str, Any] = {}

        garch = GARCHModel(model_type="garch")
        garch.fit(returns)
        models_to_fit["GARCH"] = garch

        egarch = GARCHModel(model_type="egarch")
        egarch.fit(returns)
        models_to_fit["EGARCH"] = egarch

        gjr = GARCHModel(model_type="gjr")
        gjr.fit(returns)
        models_to_fit["GJR"] = gjr

        ewma = EWMAModel()
        ewma.fit(returns)
        models_to_fit["EWMA"] = ewma

        figarch = FIGARCHModel()
        figarch.fit(returns)
        models_to_fit["FIGARCH"] = figarch

        # Phase 6: Add 4 modern models
        from src.models.har_rv import HARRVModel
        from src.models.stochastic_vol import StochasticVolModel
        from src.models.gas_volatility import GASVolModel
        from src.models.ms_garch import MSGARCHModel

        har_rv = HARRVModel()
        har_rv.fit(returns)
        models_to_fit["HAR-RV"] = har_rv

        sv = StochasticVolModel(n_advi_steps=2000, n_samples=100)
        sv.fit(returns)
        models_to_fit["StochasticVol"] = sv

        gas = GASVolModel(dist="studentst")
        gas.fit(returns)
        models_to_fit["GAS"] = gas

        ms_garch = MSGARCHModel(n_regimes=2)
        ms_garch.fit(returns)
        models_to_fit["MS-GARCH"] = ms_garch

        # Run tournament
        tournament = ModelTournament()
        for name, m in models_to_fit.items():
            tournament.add_model(name, m)
        df_results = tournament.run()

        # Build response rows
        rows: list[TournamentRow] = []
        for idx, row_data in df_results.iterrows():
            rows.append(TournamentRow(
                model_name=str(idx),
                model_type=str(row_data.get("model_type", "")),
                aic=_safe_float(row_data["aic"]) if pd.notna(row_data.get("aic")) else None,
                bic=_safe_float(row_data["bic"]) if pd.notna(row_data.get("bic")) else None,
                converged=bool(row_data.get("converged", True)),
                persistence=_safe_float(row_data["persistence"]) if pd.notna(row_data.get("persistence")) else None,
                has_arch_effects=bool(row_data["has_arch_effects"]) if row_data.get("has_arch_effects") is not None else None,
                is_normal=bool(row_data["is_normal"]) if row_data.get("is_normal") is not None else None,
                lb_pvalue=_safe_float(row_data["lb_pvalue"]) if pd.notna(row_data.get("lb_pvalue")) else None,
                arch_pvalue=_safe_float(row_data["arch_pvalue"]) if pd.notna(row_data.get("arch_pvalue")) else None,
                jb_pvalue=_safe_float(row_data["jb_pvalue"]) if pd.notna(row_data.get("jb_pvalue")) else None,
            ))

        # Determine winners
        winner_aic: str | None = None
        winner_bic: str | None = None
        try:
            winner_aic = tournament.winner("aic")
        except (ValueError, Exception):
            pass
        try:
            winner_bic = tournament.winner("bic")
        except (ValueError, Exception):
            pass

        return TournamentResponse(
            models=rows,
            winner_aic=winner_aic,
            winner_bic=winner_bic,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---- 3. FIGARCH-specific (must be before {name}/detail) ----


@router.get("/models/figarch", response_model=FigarchResponse)
async def figarch_detail() -> FigarchResponse:
    """FIGARCH model with fractional differencing parameter d."""
    try:
        returns = _get_returns()
        from src.models.figarch import FIGARCHModel

        model = FIGARCHModel()
        model.fit(returns)
        r = model.result

        diag_info = _build_diagnostics_info(model)

        return FigarchResponse(
            model_name=r.model_name,
            d=_safe_float(r.params.get("d", 0.0)),
            params={k: _safe_float(v) for k, v in r.params.items()},
            aic=_safe_float(r.aic) if r.aic is not None else None,
            bic=_safe_float(r.bic) if r.bic is not None else None,
            converged=bool(r.converged),
            persistence=_safe_float(r.persistence) if r.persistence is not None else None,
            conditional_volatility=_series_to_timepoints(r.conditional_volatility),
            diagnostics=diag_info,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---- 4. Custom model fit (must be before {name}/detail) ----


@router.post("/models/fit-custom", response_model=ModelDetailResponse)
async def fit_custom_model(req: CustomFitRequest) -> ModelDetailResponse:
    """Fit a custom GARCH-family model with user-specified parameters."""
    try:
        returns = _get_returns()
        from src.models.garch import GARCHModel

        model = GARCHModel(
            model_type=req.model_type,  # type: ignore[arg-type]
            p=req.p,
            q=req.q,
            dist=req.dist,
        )
        model.fit(returns)
        return _build_model_detail(model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---- 5. Model detail by name ----


@router.get("/models/{name}/detail", response_model=ModelDetailResponse)
async def model_detail(
    name: str = Path(..., pattern="^(garch|egarch|gjr|ewma|figarch)$"),
) -> ModelDetailResponse:
    """Model detail with conditional volatility and standardized residuals."""
    try:
        returns = _get_returns()

        if name == "ewma":
            from src.models.ewma import EWMAModel
            model: Any = EWMAModel()
        elif name == "figarch":
            from src.models.figarch import FIGARCHModel
            model = FIGARCHModel()
        else:
            from src.models.garch import GARCHModel
            model = GARCHModel(model_type=name)  # type: ignore[arg-type]

        model.fit(returns)
        return _build_model_detail(model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---- 6. EVT detailed analysis ----


@router.get("/risk/evt", response_model=EvtResponse)
async def risk_evt(
    confidence: float = Query(default=0.99, ge=0.9, le=0.999),
    percentile: float = Query(default=0.10, ge=0.01, le=0.50),
) -> EvtResponse:
    """Extreme Value Theory analysis with Hill estimator and mean excess plot."""
    try:
        returns = _get_returns()
        from src.risk.evt import EVTAnalyzer

        evt = EVTAnalyzer()
        evt.fit(returns, confidence)

        # Hill estimator at requested percentile
        hill_result = evt.hill_estimator(k_percentile=percentile)

        # Hill estimates at multiple k percentiles for Hill plot
        estimates: list[HillEstimate] = []
        for kp in [0.05, 0.10, 0.15, 0.20, 0.25]:
            try:
                h = evt.hill_estimator(k_percentile=kp)
                estimates.append(HillEstimate(
                    k_percentile=kp,
                    tail_index=_safe_float(h["tail_index"]),
                    shape=_safe_float(h["shape"]),
                ))
            except Exception:
                pass

        hill_info = HillInfo(
            tail_index=_safe_float(hill_result["tail_index"]),
            shape=_safe_float(hill_result["shape"]),
            threshold=_safe_float(hill_result["threshold"]),
            k=int(hill_result["k"]),
            estimates=estimates,
        )

        # Mean excess data
        me_df = evt.mean_excess_data()
        mean_excess: list[MeanExcessPoint] = [
            MeanExcessPoint(
                threshold=_safe_float(row["threshold"]),
                mean_excess=_safe_float(row["mean_excess"]),
            )
            for _, row in me_df.iterrows()
        ]

        # GPD parameters from the fitted result
        r = evt.result
        gpd_params = GpdParams(
            xi=_safe_float(r.gpd_shape) if r.gpd_shape is not None else 0.0,
            sigma=_safe_float(r.gpd_scale) if r.gpd_scale is not None else 0.0,
        )

        return EvtResponse(
            hill=hill_info,
            mean_excess=mean_excess,
            gpd_params=gpd_params,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---- 7. VaR backtest ----


@router.get("/risk/backtest", response_model=BacktestResponse)
async def risk_backtest(
    confidence: float = Query(default=0.99, ge=0.9, le=0.999),
    window: int = Query(default=252, ge=50, le=1000),
) -> BacktestResponse:
    """VaR backtest with Kupiec and Christoffersen coverage tests."""
    try:
        returns = _get_returns()
        from src.risk.backtest import VaRBacktest
        from src.risk.var_engine import VaREngine

        # Compute rolling VaR
        var_df = VaREngine.rolling_var(returns, window=window, confidence=confidence)
        var_series = var_df["var"]

        # Align returns to var_series index
        aligned_returns = returns.loc[var_series.index]

        # Run backtest
        bt = VaRBacktest()
        result = bt.full_backtest(aligned_returns, var_series, confidence)

        # Build var_series time points
        var_points = _series_to_timepoints(var_series)

        expected_rate = 1.0 - confidence
        actual_rate = result.n_violations / result.n_observations if result.n_observations > 0 else 0.0

        return BacktestResponse(
            violations=result.n_violations,
            n_observations=result.n_observations,
            expected_violations=_safe_float(result.expected_violations),
            actual_coverage=_safe_float(result.actual_coverage),
            passes=bool(result.passes),
            kupiec=KupiecTest(
                statistic=_safe_float(result.kupiec_stat),
                pvalue=_safe_float(result.kupiec_pvalue),
                expected_rate=_safe_float(expected_rate),
                actual_rate=_safe_float(actual_rate),
            ),
            christoffersen=ChristoffersenTest(
                statistic=_safe_float(result.christoffersen_stat),
                pvalue=_safe_float(result.christoffersen_pvalue),
            ),
            var_series=var_points,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---- 8. HMM regime detection ----


@router.get("/regimes/hmm", response_model=HmmResponse)
async def hmm_regimes(
    n_regimes: int = Query(default=3, ge=2, le=6),
) -> HmmResponse:
    """Hidden Markov Model regime detection on conditional volatility."""
    try:
        returns = _get_returns()

        # Fit GARCH to get conditional volatility
        from src.models.garch import GARCHModel
        from src.regime.hmm_regime import HMMRegimeDetector

        garch = GARCHModel(model_type="garch")
        garch.fit(returns)
        cond_vol = garch.result.conditional_volatility

        # Detect regimes
        detector = HMMRegimeDetector(n_regimes=n_regimes)
        regime_result = detector.fit(cond_vol)

        # Build labels (date, regime) pairs
        # cond_vol.index is integer-based; use original df dates for proper date strings
        labels: list[RegimeLabel] = []
        df = _get_data()
        date_values = df["date"].values if "date" in df.columns else None
        cv_indices = cond_vol.index  # integer positions (1-based from returns)
        for j, label_val in enumerate(regime_result.labels):
            if j < len(cv_indices):
                idx = int(cv_indices[j])  # integer position in original df
                if date_values is not None and 0 <= idx < len(date_values):
                    date_str = str(date_values[idx])[:10]  # "YYYY-MM-DD"
                elif hasattr(cv_indices[j], "date"):
                    date_str = str(cv_indices[j].date())
                else:
                    date_str = str(idx)
                labels.append(RegimeLabel(date=date_str, regime=int(label_val)))

        # Transition matrix as list of lists
        trans_matrix: list[list[float]] = []
        for row in regime_result.transition_matrix:
            trans_matrix.append([_safe_float(v) for v in row])

        # Per-regime statistics
        regime_stats: list[RegimeStats] = []
        for i in range(regime_result.n_regimes):
            regime_stats.append(RegimeStats(
                regime=i,
                mean=_safe_float(regime_result.regime_means.get(i, 0.0)),
                std=_safe_float(regime_result.regime_stds.get(i, 0.0)),
            ))

        return HmmResponse(
            n_regimes=regime_result.n_regimes,
            labels=labels,
            transition_matrix=trans_matrix,
            regime_stats=regime_stats,
            current_regime=int(regime_result.current_regime),
            current_regime_name=str(regime_result.current_regime_name),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---- 9. Stress test ----


@router.post("/scenarios/stress", response_model=StressResponse)
async def stress_test(req: StressRequest) -> StressResponse:
    """Stress test with elevated volatility scenarios."""
    try:
        returns = _get_returns()
        from src.analysis.scenarios import ScenarioGenerator

        gen = ScenarioGenerator.from_data(returns)
        results = gen.stress_test(
            current_spread=req.current,
            shock_multipliers=req.shock_multipliers,
            horizon=req.horizon,
            n_paths=req.n_paths,
        )

        scenarios: list[StressScenario] = []
        for name, data in results.items():
            scenarios.append(StressScenario(
                name=str(name),
                vol_multiplier=_safe_float(data.get("vol_multiplier", 1.0)),
                median_final=_safe_float(data.get("median_final", 0.0)),
                p5=_safe_float(data.get("p5_final", 0.0)),
                p95=_safe_float(data.get("p95_final", 0.0)),
                prob_exceed=_safe_float(data.get("probability_exceed_threshold", 0.0)),
            ))

        return StressResponse(scenarios=scenarios)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---- 10. Sensitivity analysis ----


@router.get("/analysis/sensitivity", response_model=SensitivityResponse)
async def sensitivity_analysis() -> SensitivityResponse:
    """Sensitivity analysis (tornado diagram) for GARCH parameters."""
    try:
        returns = _get_returns()

        from src.analysis.sensitivity import SensitivityAnalyzer
        from src.models.garch import GARCHModel

        # Fit base GARCH model to get parameters
        garch = GARCHModel(model_type="garch")
        garch.fit(returns)
        base_params = {k: float(v) for k, v in garch.result.params.items()}

        # Define metric: unconditional variance = omega / (1 - alpha - beta)
        def garch_unconditional_var(params: dict[str, float]) -> float:
            omega = params.get("omega", 0.0)
            alpha = params.get("alpha", 0.0)
            beta = params.get("beta", 0.0)
            persistence = alpha + beta
            if persistence >= 0.9999:
                return 1e6
            return omega / max(1.0 - persistence, 1e-8)

        base_value = garch_unconditional_var(base_params)

        # Build parameter ranges (±20% perturbation)
        param_ranges: dict[str, tuple[float, float]] = {}
        skippable = {"mu", "nu"}  # skip mean and degrees of freedom
        for name, val in base_params.items():
            if name in skippable or val == 0:
                continue
            param_ranges[name] = (val * 0.8, val * 1.2)

        analyzer = SensitivityAnalyzer(base_params)
        tornado_df = analyzer.tornado_diagram(param_ranges, garch_unconditional_var)

        variables: list[SensitivityVariable] = []
        for _, row in tornado_df.iterrows():
            low_val = _safe_float(row["low"])
            high_val = _safe_float(row["high"])
            base_val = _safe_float(row["base"])
            impact = abs(high_val - low_val)
            sensitivity_pct = (impact / abs(base_val) * 100.0) if base_val != 0 else 0.0

            variables.append(SensitivityVariable(
                name=str(row["param_name"]),
                low=low_val,
                high=high_val,
                base=base_val,
                sensitivity_pct=_safe_float(sensitivity_pct),
            ))

        return SensitivityResponse(
            base_value=_safe_float(base_value),
            variables=variables,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---- 11. Kalman Signal Extraction ----


@router.get("/regimes/kalman-signal", response_model=KalmanSignalResponse)
async def kalman_signal(
    column: str = Query(default="spread_all", description="Spread column to analyse"),
) -> KalmanSignalResponse:
    """Kalman filter signal extraction (Local Level Model)."""
    try:
        df = _get_data()
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index)

        if column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column}' not found")

        spread = df[column].dropna()

        from src.models.kalman import KalmanSignalExtractor

        extractor = KalmanSignalExtractor()
        extractor.fit(spread)
        result = extractor.result

        # Retrieve fitted parameters from the internal model
        fitted = extractor._fitted_model
        sigma2_eta = _safe_float(fitted.params[0])
        sigma2_eps = _safe_float(fitted.params[1])
        q_ratio = sigma2_eta / max(sigma2_eps, 1e-12)

        return KalmanSignalResponse(
            signal=_series_to_timepoints(result.signal),
            deviation=_series_to_timepoints(result.deviation),
            deviation_zscore=_series_to_timepoints(result.deviation_zscore),
            signal_strength=_safe_float(result.signal_strength),
            is_overvalued=bool(result.is_overvalued),
            is_undervalued=bool(result.is_undervalued),
            sigma2_eta=sigma2_eta,
            sigma2_eps=sigma2_eps,
            q_ratio=_safe_float(q_ratio),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---- 12. Change Point Detection ----


@router.get("/regimes/changepoints", response_model=ChangepointResponse)
async def changepoints(
    column: str = Query(default="spread_all", description="Spread column to analyse"),
    method: str = Query(default="binseg", pattern="^(pelt|binseg)$"),
    n_bkps: int = Query(default=5, ge=2, le=15, description="Number of breakpoints (binseg only)"),
    pen: float = Query(default=10.0, gt=0, description="Penalty for PELT"),
) -> ChangepointResponse:
    """Structural change point detection via PELT or Binary Segmentation."""
    try:
        import ruptures as rpt

        df = _get_data()
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index)

        if column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column}' not found")

        spread = df[column].dropna()
        signal = spread.values.astype("float64")
        dates = spread.index

        if method == "pelt":
            algo = rpt.Pelt(model="rbf", min_size=30).fit(signal)
            bkps = algo.predict(pen=pen)
        else:
            algo = rpt.Binseg(model="l2", min_size=60).fit(signal)
            bkps = algo.predict(n_bkps=n_bkps)

        # bkps always ends with len(signal)
        breakpoints = [b for b in bkps if b < len(signal)]
        breakpoint_dates = [str(dates[b].date()) for b in breakpoints]

        # Build segments
        edges = [0] + bkps
        segments: list[ChangepointSegment] = []
        for i in range(len(edges) - 1):
            seg_data = signal[edges[i]:edges[i + 1]]
            start_date = str(dates[edges[i]].date()) if edges[i] < len(dates) else str(dates[-1].date())
            end_idx = edges[i + 1] - 1
            end_date = str(dates[min(end_idx, len(dates) - 1)].date())
            segments.append(ChangepointSegment(
                start_idx=int(edges[i]),
                end_idx=int(edges[i + 1]),
                start_date=start_date,
                end_date=end_date,
                mean=_safe_float(float(seg_data.mean())),
                std=_safe_float(float(seg_data.std())) if len(seg_data) > 1 else 0.0,
            ))

        return ChangepointResponse(
            method=str(method),
            breakpoints=breakpoints,
            breakpoint_dates=breakpoint_dates,
            segments=segments,
            n_segments=len(segments),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---- 13. STS Signal Extraction ----


@router.get("/regimes/sts-signal", response_model=STSSignalResponse)
async def sts_signal(
    column: str = Query(default="spread_all", description="Spread column to analyse"),
) -> STSSignalResponse:
    """Structural Time Series signal extraction (Local Linear Trend)."""
    try:
        df = _get_data()
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index)

        if column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column}' not found")

        spread = df[column].dropna()

        from src.models.sts import STSSignalExtractor

        extractor = STSSignalExtractor()
        extractor.fit(spread)
        result = extractor.result

        return STSSignalResponse(
            signal=_series_to_timepoints(result.signal),
            deviation=_series_to_timepoints(result.deviation),
            deviation_zscore=_series_to_timepoints(result.deviation_zscore),
            signal_strength=_safe_float(result.signal_strength),
            is_overvalued=bool(result.is_overvalued),
            is_undervalued=bool(result.is_undervalued),
            level=_series_to_timepoints(result.level),
            slope=_series_to_timepoints(result.slope),
            irregular=_series_to_timepoints(result.irregular),
            aic=_safe_float(result.aic),
            bic=_safe_float(result.bic),
            n_params=int(result.n_params),
            sigma2_level=_safe_float(result.sigma2_level),
            sigma2_trend=_safe_float(result.sigma2_trend),
            sigma2_irregular=_safe_float(result.sigma2_irregular),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---- 14. Bayesian STS ----


@router.get("/regimes/bayesian-sts", response_model=BayesianSTSResponse)
async def bayesian_sts(
    column: str = Query(default="spread_all", description="Spread column to analyse"),
    n_advi_iter: int = Query(default=15000, ge=1000, le=50000, description="ADVI iterations"),
    n_samples: int = Query(default=300, ge=50, le=1000, description="Posterior samples"),
) -> BayesianSTSResponse:
    """Bayesian Structural Time Series via PyMC ADVI."""
    try:
        df = _get_data()
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index)

        if column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column}' not found")

        spread = df[column].dropna()

        from src.models.bayesian_sts import BayesianSTSSignalExtractor

        extractor = BayesianSTSSignalExtractor(
            n_advi_steps=n_advi_iter,
            n_samples=n_samples,
        )
        extractor.fit(spread)
        result = extractor.result

        return BayesianSTSResponse(
            signal=_series_to_timepoints(result.signal),
            deviation=_series_to_timepoints(result.deviation),
            deviation_zscore=_series_to_timepoints(result.deviation_zscore),
            signal_strength=_safe_float(result.signal_strength),
            is_overvalued=bool(result.is_overvalued),
            is_undervalued=bool(result.is_undervalued),
            signal_lower=_series_to_timepoints(result.signal_lower),
            signal_upper=_series_to_timepoints(result.signal_upper),
            ci_width_mean=_safe_float(result.ci_width_mean),
            sigma_level_mean=_safe_float(result.sigma_level_mean),
            sigma_obs_mean=_safe_float(result.sigma_obs_mean),
            n_samples=int(result.n_samples),
            fitting_time_sec=_safe_float(result.fitting_time_sec),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---- 15. HAR-RV Volatility ----


@router.get("/volatility/har-rv")
async def har_rv_volatility(
    column: str = Query(default="spread_all", description="Spread column for returns"),
) -> dict:
    """Heterogeneous Autoregressive Realized Volatility (Corsi 2009)."""
    try:
        from api.schemas import HARRVResponse

        returns = _get_returns()

        from src.models.har_rv import HARRVModel

        model = HARRVModel()
        model.fit(returns)
        result = model.result

        diag_info = _build_diagnostics_info(model)

        response = HARRVResponse(
            model_name=result.model_name,
            params={k: _safe_float(v) for k, v in result.params.items()},
            r_squared=_safe_float(result.params.get("r_squared", 0.0)),
            aic=_safe_float(result.aic) if result.aic is not None else None,
            bic=_safe_float(result.bic) if result.bic is not None else None,
            conditional_volatility=_series_to_timepoints(result.conditional_volatility),
            rv_daily=_series_to_timepoints(model._rv_daily),
            rv_weekly=_series_to_timepoints(model._rv_weekly),
            rv_monthly=_series_to_timepoints(model._rv_monthly),
            diagnostics=diag_info,
        )
        return response.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---- 16. Stochastic Volatility ----


@router.get("/volatility/stochastic-vol")
async def stochastic_vol(
    column: str = Query(default="spread_all", description="Spread column for returns"),
    n_advi_steps: int = Query(default=2000, ge=1000, le=50000),
    n_samples: int = Query(default=100, ge=50, le=500),
) -> dict:
    """Bayesian Stochastic Volatility model (Taylor 1986 / Kim-Shephard-Chib 1998)."""
    try:
        from api.schemas import StochasticVolResponse

        returns = _get_returns()

        from src.models.stochastic_vol import StochasticVolModel

        model = StochasticVolModel(
            n_advi_steps=n_advi_steps,
            n_samples=n_samples,
        )
        model.fit(returns)
        result = model.result

        diag_info = _build_diagnostics_info(model)

        response = StochasticVolResponse(
            model_name=result.model_name,
            params={k: _safe_float(v) for k, v in result.params.items()},
            aic=_safe_float(result.aic) if result.aic is not None else None,
            bic=_safe_float(result.bic) if result.bic is not None else None,
            conditional_volatility=_series_to_timepoints(result.conditional_volatility),
            vol_lower=_series_to_timepoints(model._vol_lower),
            vol_upper=_series_to_timepoints(model._vol_upper),
            log_vol=_series_to_timepoints(model._log_vol_mean),
            fitting_time_sec=_safe_float(model._fitting_time),
            diagnostics=diag_info,
        )
        return response.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---- 17. GAS Volatility ----


@router.get("/volatility/gas")
async def gas_volatility(
    column: str = Query(default="spread_all", description="Spread column for returns"),
    dist: str = Query(default="studentst", pattern="^(normal|studentst)$"),
) -> dict:
    """Generalized Autoregressive Score (GAS) volatility model (Creal et al. 2013)."""
    try:
        from api.schemas import GASResponse

        returns = _get_returns()

        from src.models.gas_volatility import GASVolModel

        model = GASVolModel(dist=dist)
        model.fit(returns)
        result = model.result

        diag_info = _build_diagnostics_info(model)

        response = GASResponse(
            model_name=result.model_name,
            dist=dist,
            params={k: _safe_float(v) for k, v in result.params.items()},
            aic=_safe_float(result.aic) if result.aic is not None else None,
            bic=_safe_float(result.bic) if result.bic is not None else None,
            conditional_volatility=_series_to_timepoints(result.conditional_volatility),
            score_series=_series_to_timepoints(model._score_series),
            diagnostics=diag_info,
        )
        return response.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---- 18. MS-GARCH ----


@router.get("/volatility/ms-garch")
async def ms_garch(
    column: str = Query(default="spread_all", description="Spread column for returns"),
    n_regimes: int = Query(default=2, ge=2, le=4),
) -> dict:
    """Markov-Switching GARCH model."""
    try:
        from api.schemas import MSGARCHResponse, MSRegimeInfo

        returns = _get_returns()

        from src.models.ms_garch import MSGARCHModel

        model = MSGARCHModel(n_regimes=n_regimes)
        model.fit(returns)
        result = model.result

        diag_info = _build_diagnostics_info(model)

        # Build regime info
        regime_infos = []
        for k in range(model.n_regimes):
            rp = model._regime_params[k]
            regime_infos.append(MSRegimeInfo(
                regime=k,
                omega=_safe_float(rp["omega"]),
                alpha=_safe_float(rp["alpha"]),
                beta=_safe_float(rp["beta"]),
                persistence=_safe_float(rp["alpha"] + rp["beta"]),
                mean_abs_return=_safe_float(rp["regime_mean_abs"]),
            ))

        # Build regime labels as TimePoints
        regime_labels = _series_to_timepoints(model._regime_labels.astype(float))

        # Build regime probs as list of dicts
        regime_probs = []
        for _, row in model._regime_probs.iterrows():
            regime_probs.append({k: _safe_float(v) for k, v in row.items()})

        response = MSGARCHResponse(
            model_name=result.model_name,
            n_regimes=model.n_regimes,
            params={k: _safe_float(v) for k, v in result.params.items()},
            aic=_safe_float(result.aic) if result.aic is not None else None,
            bic=_safe_float(result.bic) if result.bic is not None else None,
            conditional_volatility=_series_to_timepoints(result.conditional_volatility),
            regime_labels=regime_labels,
            regime_probs=regime_probs,
            regime_params=regime_infos,
            transition_matrix=model._trans_matrix.tolist(),
            current_regime=model._current_regime,
            current_regime_name=model._current_regime_name,
            diagnostics=diag_info,
        )
        return response.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
