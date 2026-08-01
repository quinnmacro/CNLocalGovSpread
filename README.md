# CN Local Gov Spread — v4.1

> Advanced econometric framework for China local government bond spread analysis, with a modern Next.js analytical platform.

**Author**: Quinn Liu · `quinn@quinnmacro.com` · [quinnmacro.com](https://quinnmacro.com)

---

## Architecture

```
CNLocalGovSpread/
├── src/                    # Core quant engine (pure Python, FROZEN — do not modify)
│   ├── core/               # Config, types, data engine, Wind client, simulator, base ABCs
│   ├── models/             # Volatility models: GARCH, FIGARCH, EWMA, Kalman, ML
│   ├── risk/               # VaR engine, EVT (GPD-POT), backtesting
│   ├── selection/          # Model tournament, diagnostics, Diebold-Mariano, MCS
│   ├── regime/             # HMM regime detection, market gauge
│   ├── analysis/           # Clustering, scenario generation, sensitivity
│   └── reporting/          # HTML/Excel/JSON report generator
├── api/                    # FastAPI REST layer (21 endpoints)
│   ├── app.py              # Application factory + mock data generator
│   ├── routes.py           # /api/v1/* endpoints (1060 lines)
│   └── schemas.py          # 30 Pydantic v2 response models
├── frontend/               # Next.js 16 analytical platform (TypeScript strict)
│   ├── app/analysis/       # 5 analysis pages (overview, volatility, risk, regimes, scenarios)
│   ├── components/         # 20 chart + 9 narrative + 4 interactive + 21 UI components
│   ├── hooks/              # 20 TanStack Query hooks + scroll-spy
│   └── lib/                # API client + TypeScript types + utilities
├── tests/                  # pytest test suite (53 tests, FROZEN)
│   ├── unit/               # Per-module unit tests
│   ├── integration/        # End-to-end pipeline tests
│   └── validation/         # Statistical validation tests
├── scripts/                # CLI entry points
│   ├── run_dashboard.py    # Dashboard launcher (legacy)
│   └── download_data.py    # Wind EDB data downloader
├── dashboard/              # Legacy Dash multi-page dashboard (pending deprecation)
├── legacy/v3.0/            # Archived previous version
└── pyproject.toml          # PEP 621 build config (hatchling)
```

## Key Features

### Quantitative Engine (`src/`)

#### Volatility Models
- **GARCH(1,1)**, **EGARCH(1,1)**, **GJR-GARCH** via `arch` library
- **FIGARCH** with GPH long-memory estimator and π-weight truncation
- **EWMA** with QLIK-optimal λ calibration (Patton 2011)
- **Kalman Filter** signal extraction (Local Level Model)
- **ML Volatility** (XGBoost / LightGBM) with walk-forward CV — no look-ahead bias

#### Risk Analysis
- **VaR**: Historical, Parametric-t, EVT-POT, Rolling window
- **EVT**: GPD-POT fitting, Hill tail index estimator, mean excess plot
- **Backtesting**: Kupiec unconditional + Christoffersen conditional coverage

#### Model Selection
- **Tournament**: AIC/BIC comparison with residual diagnostics
- **Diagnostics**: Ljung-Box, ARCH-LM, Jarque-Bera
- **Forecast Tests**: Diebold-Mariano (HAC), Model Confidence Set (Hansen 2011)

#### Regime Detection & State Space Methods
- **HMM**: GaussianHMM with regime sorting by mean volatility
- **Market Gauge**: Sigmoid-based composite stress indicator (5 dimensions)
- **Kalman Filter**: Local Level Model signal extraction with z-score deviation
- **Structural Time Series (STS)**: Level + slope decomposition via statsmodels
- **Bayesian STS**: PyMC posterior inference with 80% credible intervals
- **Change Point Detection**: PELT / Binary Segmentation via `ruptures`

#### Scenario Analysis
- **Monte Carlo**: AR(1)+GARCH(1,1) Student-t DGP
- **Stress Tests**: Volatility multiplier scenarios with tail probabilities
- **Fan Charts**: Percentile-band forward projections

### Analytical Platform (`frontend/`)

A Bloomberg Terminal × Linear aesthetic dark-theme web application built with Next.js 16, featuring:

- **5 Analysis Pages** following a WHY → HOW → WHAT → SO WHAT → NOW WHAT narrative structure
- **20 Chart Components** (Plotly, dynamically imported, SSR-safe)
- **9 Narrative Components** (KaTeX formulas, read guides, parameter tooltips, insight cards)
- **20 API Hooks** (TanStack Query v5 with automatic caching and refetch)
- **Institutional Finance Design System** (oklch colors, Inter + JetBrains Mono fonts)

#### Page Overview

| Page | Content | Key Methods |
|------|---------|-------------|
| `/analysis/overview` | Spread time series, KDE distribution, term structure | Data statistics, ADF test |
| `/analysis/volatility` | GARCH tournament, FIGARCH, residual diagnostics | GARCH/EGARCH/GJR/FIGARCH |
| `/analysis/risk` | VaR comparison, EVT analysis, backtesting | Historical/Parametric/EVT VaR |
| `/analysis/regimes` | HMM + Kalman + STS + Bayesian + CPD + MarketGauge | 6 state detection methods |
| `/analysis/scenarios` | Fan charts, Monte Carlo paths, stress tests | AR(1)+GARCH MC simulation |

## Quick Start

### Prerequisites
- Python 3.13+, Node.js 22+
- macOS / Linux (Wind integration requires macOS/Windows)

### Installation

```bash
# Clone and install Python package
git clone https://github.com/quinnmacro/CNLocalGovSpread.git
cd CNLocalGovSpread
pip install -e ".[dev]"

# Install frontend dependencies
cd frontend
npm install
```

### Data Sources

Configure via environment variable `CLS_DATA__SOURCE`:

| Source | Description |
|--------|-------------|
| `mock` | AR(1)+GARCH synthetic data (default for development) |
| `csv` | Load from local CSV file |
| `wind` | Wind Financial Terminal EDB (requires WindPy) |

```bash
export CLS_DATA__SOURCE=mock  # or csv, wind
export CLS_DATA__CSV_PATH=data/spreads.csv
```

### Run Development Servers

```bash
# Terminal 1: Backend API (FastAPI)
cd /path/to/CNLocalGovSpread
CLS_DATA__SOURCE=mock python3.13 -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Frontend (Next.js)
cd /path/to/CNLocalGovSpread/frontend
npm run dev
# → http://localhost:3000
```

### Run Tests

```bash
# Python tests (must remain 53 passed)
python3.13 -m pytest tests/ -v

# TypeScript type checking (must be 0 errors)
cd frontend && npx tsc --noEmit

# Next.js production build
cd frontend && npx next build
```

## API Reference

The REST API exposes 21 endpoints under `/api/v1/`. The frontend proxies all requests via `next.config.ts` rewrites.

### Endpoint Groups

| Group | Endpoints | Description |
|-------|-----------|-------------|
| **Data** | `/health`, `/data/summary`, `/data/raw`, `/data/statistics` | Health check, data overview, raw data, statistics |
| **Models** | `/models/fit`, `/models/tournament`, `/models/{name}/detail`, `/models/figarch`, `/models/fit-custom` | Volatility model fitting and comparison |
| **Risk** | `/risk/metrics`, `/risk/evt`, `/risk/backtest` | VaR, EVT analysis, backtesting |
| **Regimes** | `/regimes/hmm`, `/regimes/kalman-signal`, `/regimes/sts-signal`, `/regimes/bayesian-sts`, `/regimes/changepoints`, `/market/gauge` | State detection (6 methods) |
| **Scenarios** | `/scenarios/generate`, `/scenarios/stress`, `/analysis/sensitivity` | Monte Carlo, stress tests, sensitivity |

Full endpoint details with request/response schemas: see [`HANDOFF-v4.1.md`](HANDOFF-v4.1.md) §2.

## Configuration

All settings use Pydantic v2 `BaseSettings` with environment variable prefix `CLS_`:

| Variable | Default | Description |
|---|---|---|
| `CLS_DATA__SOURCE` | `csv` | Data source: mock, csv, wind |
| `CLS_DATA__CSV_PATH` | `data/spreads.csv` | CSV file path |
| `CLS_DATA__START_DATE` | `2018-01-01` | Start date filter |
| `CLS_RISK__CONFIDENCE` | `0.99` | VaR confidence level |
| `CLS_RISK__HORIZON` | `252` | Forecast horizon (days) |
| `CLS_DASHBOARD__PORT` | `8050` | Legacy dashboard port |

## Design Principles

1. **ABC interfaces** — `VolatilityModel`, `SignalExtractor`, `RiskAnalyzer` for polymorphism
2. **Frozen dataclasses** — `VolatilityResult`, `SignalResult`, `RiskResult`, `RegimeResult`
3. **No data leakage** — ML targets are `r²[t+1]`, features use only `r²[t-k]`
4. **Structured logging** — no `print()`, always `get_logger(__name__)`
5. **Graceful degradation** — optional deps (XGBoost, WindPy, hmmlearn) with clean fallbacks
6. **Self-contained reports** — HTML with embedded Plotly, no CDN dependency
7. **Narrative structure** — every analysis page follows WHY → HOW → WHAT → SO WHAT → NOW WHAT
8. **TypeScript strict** — no `any`, all API responses fully typed

## Wind 数据集成

项目包含完整的 Wind Financial Terminal 数据集成模块，支持地方债信用利差 EDB 数据的自动下载和清洗。

### WindClient 模块

```python
from src.core.wind_client import WindClient, DEFAULT_SPREAD_CODES

# 自动检测 Wind 路径（macOS/Windows）
with WindClient() as client:
    df = client.fetch_edb(
        codes=DEFAULT_SPREAD_CODES,
        start_date="2018-01-01",
        end_date="2026-08-01",
        fill_method="Previous",
    )
```

**特性**:
- ✅ 自动检测 macOS/Windows Wind Python API 路径
- ✅ 连接生命周期管理（context manager）
- ✅ 失败自动重试（最多 2 次）
- ✅ 增量更新支持（检测已有 CSV 最新日期）
- ✅ Wind 异常值清洗（-999, 999, -9999 占位符）

### EDB 指标代码

| 代码 | 含义 | DataFrame 列名 |
|------|------|----------------|
| M0017142 | 地方债信用利差综合 | spread_all |
| M0017143 | 地方债 5Y 信用利差 | spread_5y |
| M0017144 | 地方债 10Y 信用利差 | spread_10y |
| M0017145 | 地方债 30Y 信用利差 | spread_30y |

### 下载脚本

```bash
# 全量下载（2018 至今）
python scripts/download_data.py

# 增量更新
python scripts/download_data.py --incremental

# 指定日期范围
python scripts/download_data.py --start 2024-01-01 --end 2026-08-01
```

## Documentation

| Document | Description |
|----------|-------------|
| [`HANDOFF-v4.1.md`](HANDOFF-v4.1.md) | **Current handoff** — complete project state, API catalog, component reference |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history from v1.0 to v4.1 |
| [`frontend/README.md`](frontend/README.md) | Frontend development guide and component catalog |
| [`HANDOFF-v4.md`](HANDOFF-v4.md) | Previous handoff (Phase 1–2, superseded) |

## License

MIT © 2026 QuinnMacro
