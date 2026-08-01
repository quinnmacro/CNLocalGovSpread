# CN Local Gov Spread — v4.0

> Advanced econometric framework for China local government bond spread analysis.

**Author**: Quinn Liu · `quinn@quinnmacro.com` · [quinnmacro.com](https://quinnmacro.com)

---

## Architecture

```
CNLocalGovSpread/
├── src/                    # Core quant engine (pure Python, no UI)
│   ├── core/               # Config, types, data engine, Wind client, simulator, base ABCs
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
├── tests/                  # pytest test suite (53 tests)
│   ├── unit/               # Per-module unit tests
│   ├── integration/        # End-to-end pipeline tests
│   └── validation/         # Statistical validation tests
├── scripts/                # CLI entry points
│   ├── run_dashboard.py    # Dashboard launcher
│   └── download_data.py    # Wind EDB data downloader
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

### 下载脚本

```bash
# 全量下载（2018 至今）
python scripts/download_data.py

# 指定日期范围
python scripts/download_data.py --start 2024-01-01 --end 2026-08-01

# 增量更新（检测 data/local_gov_spread.csv 最新日期）
python scripts/download_data.py --incremental

# 同时下载信用利差对比数据（需配置 CREDIT_SPREAD_CODES）
python scripts/download_data.py --credit

# 自定义 Wind 路径
python scripts/download_data.py --wind-path "/custom/path/to/wind"
```

### EDB 指标代码

| 代码 | 含义 | DataFrame 列名 |
|------|------|----------------|
| M0017142 | 地方债信用利差综合 | spread_all |
| M0017143 | 地方债 5Y 信用利差 | spread_5y |
| M0017144 | 地方债 10Y 信用利差 | spread_10y |
| M0017145 | 地方债 30Y 信用利差 | spread_30y |

**信用利差对比指标**（企业债/中票 AAA 各期限）需要在 `src/core/wind_client.py` 中填入实际 Wind EDB 代码：

```python
CREDIT_SPREAD_CODES = {
    "credit_corp_aaa_3y":  "M00XXXXX",  # 企业债AAA 3Y
    "credit_corp_aaa_5y":  "M00XXXXX",  # 企业债AAA 5Y
    # ... 更多指标
}
```

### 数据流

```
Wind EDB → WindClient.fetch_edb() → DataEngine.load() → DataEngine.clean()
  ↓                                                              ↓
原始数据                                                   MAD 异常值清洗
                                                        + 前向填充 (ffill)
                                                        + 后向填充 (bfill)
```

### 环境要求

- **macOS**: Wind Financial Terminal + Python API (`/Applications/Wind API.app/Contents/python`)
- **Windows**: Wind.NET Client + Python API (`C:\Wind\Wind.NET.Client\WindNET\x64`)
- **Linux**: 不支持（Wind 仅提供 macOS/Windows 版本）

## License

MIT © 2026 QuinnMacro
