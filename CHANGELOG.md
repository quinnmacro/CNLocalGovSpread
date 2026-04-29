# Changelog

All notable changes to the CNLocalGovSpread project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-04-29

### Phase 1: Algorithm Depth

#### Fixed
- **Hill estimator abs(xi) bug**: Changed 1/abs(xi) to -1/xi when xi<0, correctly indicating short-tailed distribution instead of misleading positive tail index
- **GARCH convergence checking**: Added convergence_flag validation (0=converged) for all GARCH models (GARCH, EGARCH, GJR-GARCH), storing converged status in ic_scores dict
- **Parameter diagnostics extraction**: Added `get_parameter_diagnostics()` method to VolatilityModeler that extracts parameter estimates, std errors, t-statistics, p-values, and significance flags from arch model results
- **Mean Excess Plot for EVT threshold selection**: Added `mean_excess_plot()` method to EVTRiskAnalyzer with Plotly interactive visualization, automatic optimal threshold detection via MEF stability analysis, and current threshold marking
- **EGARCH gamma check remnant bug in report.py**: Replaced misleading gamma[1] extraction (always 0 since arch EGARCH doesn't have that parameter) with proper logic that references GJR-GARCH results for explicit asymmetry measurement

#### Added
- **ML volatility model comparison module** (`src/ml_volatility.py`): MLVolatilityModeler class implementing Random Forest, XGBoost, and LSTM volatility models competing alongside GARCH tournament with unified AIC/BIC + RMSE/MAE evaluation
- **Parameter Auto-Calibration module** (`src/calibration.py`): ParameterCalibrator class that estimates 6 model parameters from real data: EWMA lambda (QLIKE optimization), t-distribution df (MLE), AR(1) phi (OLS), EVT threshold percentile (MEF stability), Kalman fallback window (cross-validation), and signal deviation threshold (quantile estimation)
- **FIGARCH long-memory detection** (`src/volatility.py`): `detect_long_memory()` method using GPH (Geweke-Porter-Hudak) semi-parametric estimator for fractional differencing parameter d estimation, plus `fit_figarch()` method implementing FIGARCH(1,d,1) via truncated polynomial expansion with numerical optimization
- **Calibration pipeline integration**: Wired calibrated parameters into the analysis pipeline (EWMA lambda→GARCH, EVT threshold→risk analysis, Kalman window→fallback) making calibration functionally impactful rather than purely decorative
- **Phase 1 v3.0 modules integrated into dashboard**: MLVolatilityModeler, ParameterCalibrator, and FIGARCH long-memory detection now appear in 波动率分析 page (FIGARCH长记忆检测, ML波动率模型对比, 参数自动校准 sections)

### Phase 2: Dashboard 2.0

#### Changed
- **Dashboard refactored into multi-page Streamlit architecture**: Split single 684-line dashboard.py into app.py home page, shared_state.py module for shared logic, and 6 individual page files in pages/ directory (信号分析, 波动率分析, 风险分析, 情景分析, 历史回溯, 报告中心)
- **Signal analysis page enhanced with calibration visibility** (v3.0): Added 参数校准详情 expander showing calibrated Kalman window vs default (60天) and signal deviation threshold vs default (1.5σ); trading signal threshold now uses calibrated value when available instead of hardcoded 1.5σ
- **Volatility analysis page enhanced** (v3.0): Added FIGARCH长记忆检测, ML波动率模型对比, 参数自动校准 sections with calibrated vs default parameter comparison table
- **Risk analysis page enhanced** (v3.0): Added ProvinceClusterMap choropleth + radar + cluster summary metrics alongside EVT VaR/ES analysis
- **Scenario page enhanced** (v3.0): Added MC路径可视化 and 敏感性分析 with Plotly charts

#### Added
- **MarketStatusGauge module** (`src/market_status.py`): 5-indicator composite scoring system (spread position, volatility regime, VaR breach, signal deviation, trend momentum) with weighted fusion, 5-level status classification (safe/watch/caution/warning/danger), extreme indicator upgrade logic, and 3 Plotly visualizations (multi-segment gauge, radar linkage chart, rolling timeline)
- **ProvinceClusterMap module** (`src/province_cluster.py`): 31-province hierarchical clustering using regional characteristic seeds (东部沿海/中部内陆/西部开发/东北老工业), 5-feature vectors, Ward/complete/average methods, and 3 Plotly visualizations (clustered heatmap, **true Choropleth map** with go.Choropleth + GeoJSON data, cluster comparison radar chart). Bubble map fallback when GeoJSON unavailable.
- **PPT report generation and template system** (`src/report_gen.py`): Added PowerPoint report generation via python-pptx with title, section, executive summary (for executive template), and disclaimer slides. Added 3 report templates (professional/academic/executive) with distinct color schemes and styling parameters driving HTML/CSS, PDF, and PPT formatting. Report center page now offers PPT format option and template selector.

### Phase 3: Test Coverage & Documentation

#### Added
- **Expanded test suite from 17 tests to 893 tests** across 20 test files:
  - `test_all.py` (17) - original core module tests
  - `test_ml_volatility.py` (15) - ML volatility module tests
  - `test_calibration.py` (39) - parameter calibration tests
  - `test_figarch.py` (23) - FIGARCH long-memory tests
  - `test_market_status.py` (37) - market status gauge tests
  - `test_province_cluster.py` (57) - province clustering tests (incl. Choropleth upgrade)
  - `test_alerts.py` (37) - risk alerts system tests
  - `test_scenarios.py` (56) - scenario/stress analysis tests
  - `test_visualization.py` (80) - Plotly visualization tests
  - `test_report_gen.py` (46) - report generation tests (incl. PPT + templates)
  - `test_content.py` (26) - content helper function tests
  - `test_kalman.py` (41) - KalmanSignalExtractor tests (init, fit, fallback, deviation, data patterns, integration, boundary)
  - `test_evt.py` (60) - EVTRiskAnalyzer tests (init, fit_gpd, VaR, ES, tail index, mean excess plot, Hill estimator, integration, boundary)
  - `test_data_engine.py` (59) - DataEngine tests (init, load dispatch, mock gen, CSV, clean/MAD, returns, workflow, boundary)
  - `test_styles.py` (30) - Streamlit styles rendering tests
  - `test_export.py` (16) - Excel export tests
  - `test_dashboard_integration.py` (67) - dashboard integration tests (incl. calibration pipeline + signal page)
  - `test_volatility.py` (102) - VolatilityModeler and RegimeDetector tests (init, fit_garch/egarch/gjr_garch/ewma, tournament, conditional volatility, parameter diagnostics, regime detection, model properties, boundary, integration)
  - `test_report.py` (50) - generate_strategic_report tests (model tournament output, asymmetry sections, risk status labels, volatility regime, action recommendations, boundary conditions, full pipeline integration)
  - `conftest.py` - shared pytest fixtures (15 factories + 3 mock fixtures + markers)

- **CI-friendly test runner** (`run_tests.sh`) with 6 modes: all/quick/coverage/integration/dashboard/smoke/count
- **GitHub Actions CI workflow** (`.github/workflows/ci.yml`): Automated test+lint pipeline on push/PR across Python 3.9-3.12 with test count verification (≥200)
- **reportlab dependency** added to requirements.txt for PDF generation
- **python-pptx dependency** added to requirements.txt for PPT generation
- **XGBoost dependency** added to requirements.txt (optional LSTM via commented tensorflow)
- **Version unified** to 3.0.0 across all project files (6 files: __init__.py, report_gen.py, styles.py, dashboard.py, shared_state.py, README badge)
- **CHANGELOG.md** created for tracking project evolution
- **README.md** updated with v3.0 features, new project structure, and module documentation
- **Module exports completed** in __init__.py: all 50+ public API symbols (scenarios, alerts, styles, content, export modules) accessible via `from src import`

#### Fixed
- **Plotly Heatmap ColorBar titlefont bug**: Replaced deprecated titlefont parameter with title=dict(text=..., font=...) in visualization.py
- **report_gen.py spread variable scope bug**: Moved spread definition to top of _prepare_report_data so it's available to all report sections
- **Calibration display bug on volatility page**: Replaced nested dict access (calibrated.get('key', {}).get('value', None)) with direct float access (calibrated.get('key')) since ParameterCalibrator stores plain floats not nested dicts
- **5 test collection errors**: Corrected sys.path configuration in test files using 'from src.xxx import' pattern
- **python-pptx MS_ANCHOR import**: Fixed ImportError in python-pptx v1.0.2 where MS_ANCHOR was removed from pptx.enum.text

---

## [2.3.0] - 2025-xx-xx

### Added
- 计量经济学教育内容模块 (`src/content.py`)
- 模型理论解释（Kalman/GARCH/EVT）
- 指标解读指南
- 交易建议模板
- 市场背景故事
- Streamlit仪表板交互增强

---

## [2.0.0] - 2025-xx-xx

### Added
- 主题样式系统 (`src/styles.py`)
- 情景分析模块 (`src/scenarios.py`) - 压力测试、蒙特卡洛模拟
- 风险预警系统 (`src/alerts.py`)
- 报告生成中心 (`src/report_gen.py`) - PDF/Excel/HTML
- Streamlit仪表板 (`dashboard.py`)
- 双主题（深色/浅色）切换

---

## [1.0.0] - 2024-xx-xx

### Added
- 数据引擎 (`src/data_engine.py`)
- GARCH模型锦标赛 (`src/volatility.py`) - GARCH/EGARCH/GJR-GARCH
- 卡尔曼滤波器 (`src/kalman.py`)
- 极值理论分析 (`src/evt.py`) - GPD尾部风险
- 可视化模块 (`src/visualization.py`) - Plotly交互式图表
- 战略报告生成 (`src/report.py`)
- 基础单元测试 (17个)