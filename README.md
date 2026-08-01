# CN Local Gov Spread — v4.0

> Advanced econometric framework for China local government bond spread analysis.

**Author**: Quinn Liu · `quinn@quinnmacro.com` · [quinnmacro.com](https://quinnmacro.com)

---

## Architecture

```
CNLocalGovSpread/
├── src/                    # Core quant engine (pure Python, no UI)
│   ├── core/               # Config, types, data engine, simulator, base ABCs
│   ├── models/             # Volatility models: GARCH, FIGARCH, EWMA, Kalman, ML
│   ├── risk/               # VaR engine, EVT (GPD-POT), backtesting
│   ├── selection/          # Model tournament, diagnostics, Diebold-Mariano, MCS
│   ├── regime/             # HMM regime detection, market gauge
│   ├── analysis/           # Clustering, scenario generation, sensitivity
│   └── reporting/          # HTML/Excel/JSON report generator
├── api/                    # FastAPI REST layer
│   ├── app.py              # Application factory
│   └── routes.py           # /api/v1/* endpoints
├── dashboard/              # Dash multi-page dashboard
│   ├── app.py              # Dash application factory (DARKLY theme)
│   ├── components/         # Shared charts, data cache
│   └── pages/              # 5 pages: home, volatility, risk, regimes, scenarios
├── tests/                  # pytest test suite (48 tests)
│   ├── unit/               # Per-module unit tests
│   ├── integration/        # End-to-end pipeline tests
│   └── validation/         # Statistical validation tests
├── scripts/                # CLI entry points
│   └── run_dashboard.py    # Dashboard launcher
├── legacy/v3.0/            # Archived previous version
└── pyproject.toml          # PEP 621 build config (hatchling)
```

## Key Features

### Volatility Models
- **GARCH(1,1)**, **EGARCH(1,1)**, **GJR-GARCH** via `arch` library
- **FIGARCH** with GPH long-memory estimator and π-weight truncation
- **EWMA** with QLIK-optimal λ calibration (Patton 2011)
- **Kalman Filter** signal extraction (Local Level Model)
- **ML Volatility** (XGBoost / LightGBM) with walk-forward CV — no look-ahead bias

### Risk Analysis
- **VaR**: Historical, Parametric-t, EVT-POT, Rolling window
- **EVT**: GPD-POT fitting, Hill tail index estimator, mean excess plot
- **Backtesting**: Kupiec unconditional + Christoffersen conditional coverage

### Model Selection
- **Tournament**: AIC/BIC comparison with residual diagnostics
- **Diagnostics**: Ljung-Box, ARCH-LM, Jarque-Bera
- **Forecast Tests**: Diebold-Mariano (HAC), Model Confidence Set (Hansen 2011)

### Regime Detection
- **HMM**: GaussianHMM with regime sorting by mean volatility
- **Market Gauge**: Sigmoid-based composite stress indicator (5 dimensions)

### Scenario Analysis
- **Monte Carlo**: AR(1)+GARCH(1,1) Student-t DGP
- **Stress Tests**: Volatility multiplier scenarios with tail probabilities
- **Fan Charts**: Percentile-band forward projections

## Quick Start

### Installation
```bash
# Clone and install
git clone https://github.com/quinnmacro/CNLocalGovSpread.git
cd CNLocalGovSpread
pip install -e ".[dev]"

# With ML support
pip install -e ".[ml]"

# With Wind data
pip install -e ".[wind]"
```

### Data Sources
Configure via environment variable `CLS_DATA__SOURCE`:
- `mock` — AR(1)+GARCH synthetic data (default for development)
- `csv` — load from local CSV file
- `wind` — Wind Financial Terminal EDB (requires WindPy)

```bash
export CLS_DATA__SOURCE=mock  # or csv, wind
export CLS_DATA__CSV_PATH=data/spreads.csv
```

### Run Dashboard
```bash
python scripts/run_dashboard.py --port 8050 --debug
# or
cls-dashboard
```

### Run API
```bash
uvicorn api.app:app --reload --port 8000
# or
cls-api
```

### Run Tests
```bash
pytest tests/ -v
```

## Configuration

All settings use Pydantic v2 `BaseSettings` with environment variable prefix `CLS_`:

| Variable | Default | Description |
|---|---|---|
| `CLS_DATA__SOURCE` | `csv` | Data source: mock, csv, wind |
| `CLS_DATA__CSV_PATH` | `data/spreads.csv` | CSV file path |
| `CLS_DATA__START_DATE` | `2018-01-01` | Start date filter |
| `CLS_RISK__CONFIDENCE` | `0.99` | VaR confidence level |
| `CLS_RISK__HORIZON` | `252` | Forecast horizon (days) |
| `CLS_DASHBOARD__PORT` | `8050` | Dashboard port |
| `CLS_DASHBOARD__HOST` | `127.0.0.1` | Dashboard host |

## Design Principles

1. **ABC interfaces** — `VolatilityModel`, `SignalExtractor`, `RiskAnalyzer` for polymorphism
2. **Frozen dataclasses** — `VolatilityResult`, `SignalResult`, `RiskResult`, `RegimeResult`
3. **No data leakage** — ML targets are `r²[t+1]`, features use only `r²[t-k]`
4. **Structured logging** — no `print()`, always `get_logger(__name__)`
5. **Graceful degradation** — optional deps (XGBoost, WindPy, hmmlearn) with clean fallbacks
6. **Self-contained reports** — HTML with embedded Plotly, no CDN dependency

## License

MIT © 2026 QuinnMacro
