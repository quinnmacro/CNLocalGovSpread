# 🤝 Handoff — CNLocalGovSpread v4.0 Frontend Rewrite

> **Generated**: 2026-08-01  
> **Phase 1 + Phase 2 Complete** → Next session: Phase 3–4 (analysis pages)

---

## 1. 项目状态总览

| 层级 | 状态 | 文件数 | 备注 |
|------|------|--------|------|
| `src/` (量化引擎) | ✅ 不动 | 53 tests pass | 禁止修改 |
| `api/schemas.py` | ✅ 完成 | 25 个 Pydantic model | 279 行 |
| `api/routes.py` | ✅ 完成 | 17 endpoints | 845 行 |
| `frontend/` | ✅ Phase 1 | 48 个 TS/TSX 文件 | build 0 errors |
| `dashboard/` (旧 Dash) | ⏳ 待移入 legacy | Phase 5 |

---

## 2. API 端点完整清单 (17 个)

### 原有 7 个 (未改逻辑)

| # | Method | Path | 响应 |
|---|--------|------|------|
| 1 | GET | `/api/v1/health` | `{status, version, data_source}` |
| 2 | GET | `/api/v1/data/summary` | `{n_rows, n_columns, date_range, columns, summary_stats}` |
| 3 | GET | `/api/v1/data/raw?limit=100&offset=0` | `{data[], total, offset, limit}` |
| 4 | GET | `/api/v1/models/fit?model_type=garch` | `{model_name, aic, bic, converged, params}` |
| 5 | GET | `/api/v1/risk/metrics?confidence=0.99` | `{var_historical, var_parametric, var_evt, es_evt}` |
| 6 | GET | `/api/v1/scenarios/generate?horizon=252&n_paths=5000` | `{current_spread, horizon, n_paths, median/p5/p95}` |
| 7 | GET | `/api/v1/market/gauge` | `{composite, status[], indicators{}}` |

### 新增 10 个

| # | Method | Path | 响应 (schemas.py class) | 实测数据 |
|---|--------|------|------------------------|---------|
| 8 | GET | `/api/v1/data/statistics` | `DataStatisticsResponse` | 4 columns, 含 ADF 检验 |
| 9 | GET | `/api/v1/models/tournament` | `TournamentResponse` | 5 models, winner_aic=FIGARCH |
| 10 | GET | `/api/v1/models/{name}/detail` | `ModelDetailResponse` | GARCH: 1499 vol points + residuals |
| 11 | GET | `/api/v1/models/figarch` | `FigarchResponse` | d=0.188, aic=277.3 |
| 12 | POST | `/api/v1/models/fit-custom` | `ModelDetailResponse` | EGARCH converged=true |
| 13 | GET | `/api/v1/risk/evt?percentile=0.10` | `EvtResponse` | tail_index=1.53, xi=-0.30 |
| 14 | GET | `/api/v1/risk/backtest?confidence=0.99` | `BacktestResponse` | 101 violations, passes=false |
| 15 | GET | `/api/v1/regimes/hmm?n_regimes=3` | `HmmResponse` | 3 regimes, current=0, 1499 labels |
| 16 | POST | `/api/v1/scenarios/stress` | `StressResponse` | N scenarios returned |
| 17 | GET | `/api/v1/analysis/sensitivity` | `SensitivityResponse` | base=1M, 3 variables |

---

## 3. 前端文件结构

```
frontend/
├── app/
│   ├── layout.tsx (68L)           — dark, zh, Inter+JetBrains, TooltipProvider, Navbar+Footer
│   ├── page.tsx (276L)            — Hero + LiveSnapshot + Abstract + Framework + NavCards
│   ├── globals.css (150L)         — oklch dark theme, prose-narrative, chart-container
│   ├── _components/
│   │   └── live-snapshot.tsx (88L) — 4 MetricCards from useMarketGauge/useRiskMetrics/useDataSummary
│   └── analysis/
│       ├── overview/page.tsx      — STUB (Sidebar + Breadcrumb + placeholder)
│       ├── volatility/page.tsx    — STUB
│       ├── risk/page.tsx          — STUB
│       ├── regimes/page.tsx       — STUB
│       └── scenarios/page.tsx     — STUB
├── components/
│   ├── providers.tsx (23L)        — QueryClientProvider
│   ├── narrative/
│   │   ├── section.tsx (57L)      — WHY/HOW/WHAT/SO WHAT/NOW WHAT 5-segment wrapper
│   │   ├── prose-block.tsx (18L)  — prose-narrative wrapper
│   │   ├── formula.tsx (24L)      — KaTeX block/inline
│   │   ├── read-guide.tsx (52L)   — collapsible "📖 读图指南"
│   │   ├── param-tooltip.tsx (51L) — hover tooltip for parameters
│   │   ├── insight-card.tsx (56L)  — 发现/警示 callout card
│   │   ├── metric-card.tsx (133L) — value + trend icon + mini sparkline
│   │   └── navigation-card.tsx (55L) — hover-animated page entry card
│   ├── layout/
│   │   ├── navbar.tsx (128L)      — sticky top, desktop links + mobile Sheet
│   │   ├── sidebar.tsx (88L)      — left sidebar for /analysis/* pages
│   │   ├── breadcrumb.tsx (38L)   — 面包屑
│   │   └── footer.tsx (29L)       — copyright + data source
│   ├── charts/
│   │   └── plotly-chart.tsx (106L) — dark theme Plotly, dynamic import (no SSR), responsive
│   └── ui/ (21 shadcn components) — button, card, badge, tabs, collapsible, tooltip, etc.
├── hooks/
│   ├── use-api.ts (201L)          — 20 TanStack Query hooks for all 17 endpoints
│   └── use-theme.ts (8L)          — stub (fixed dark theme)
├── lib/
│   ├── api.ts (102L)              — fetch wrapper + typed convenience methods
│   ├── types.ts (269L)            — 30+ TypeScript interfaces mirroring Pydantic schemas
│   └── utils.ts (31L)             — cn(), fmt(), fmtBps(), fmtPct(), percentileRank()
├── types/
│   └── react-katex.d.ts (13L)     — module declaration
├── next.config.ts                 — API proxy rewrite → localhost:8000
├── tsconfig.json                  — strict, paths @/* → ./*
└── package.json                   — all deps installed
```

---

## 4. 技术栈确认

```
Next.js 16.2.12 (App Router, Turbopack)   — build passes ✓
TypeScript 5.x (strict, no `any`)         — 0 TS errors ✓
Tailwind CSS 4 + shadcn/ui (base-ui)      — oklch dark theme
react-plotly.js + plotly.js 3.7           — dynamic import (no SSR)
react-katex                                — math formulas
framer-motion 12                           — page transitions, hover animations
@tanstack/react-query 5                    — server state (20 hooks)
zod 4                                      — runtime validation (available, not yet used)
date-fns 4                                 — date formatting
lucide-react                               — icons
```

### 关键依赖注意事项

- **shadcn/ui v4** 使用 `@base-ui/react`（不是 Radix），API 有些不同：
  - `TooltipProvider` 用 `delay` 而非 `delayDuration`
  - `SheetTrigger` 没有 `asChild` prop
  - `Collapsible` 支持 `open`/`onOpenChange` 控制
- **react-plotly.js** 必须用 `next/dynamic` 加载 (ssr: false)
- **react-katex** 没有官方 TypeScript 类型，用 `types/react-katex.d.ts` 声明

---

## 5. 量化引擎 Import 路径 (已验证可用)

```python
from src.core.config import DataSource, get_settings
from src.core.data_engine import DataEngine       # .load(), .compute_returns()
from src.models.garch import GARCHModel            # .fit(returns) → .result → VolatilityResult
from src.models.ewma import EWMAModel
from src.models.figarch import FIGARCHModel
from src.models.kalman import KalmanSignalExtractor  # .fit(spread Series) → .result
from src.selection.tournament import ModelTournament
from src.selection.forecast_test import diebold_mariano_test, ModelConfidenceSet
from src.risk.evt import EVTAnalyzer               # .fit(returns, conf), .hill_estimator(), .mean_excess_data()
from src.risk.backtest import VaRBacktest           # .full_backtest(returns, var_series, conf)
from src.risk.var_engine import VaREngine           # static: .historical_var(), .parametric_var(), .evt_var()
from src.regime.hmm_regime import HMMRegimeDetector # .fit(vol_series) → RegimeResult
from src.regime.market_gauge import MarketGauge     # .assess(spread, returns) → {composite, status, indicator_scores}
from src.analysis.scenarios import ScenarioGenerator # .from_data(returns), .generate(), .stress_test()
from src.analysis.sensitivity import SensitivityAnalyzer
from src.analysis.clustering import SpreadClustering
```

### Engine API 模式 (踩过坑后的结论)

```python
# VolatilityResult
result.conditional_volatility  # pd.Series (NOT .volatility)
result.persistence             # property (alpha + beta)
result.aic, result.bic, result.converged, result.params, result.model_name

# VaREngine (static methods, return dict with 'var' and 'es')
VaREngine.historical_var(returns, conf) → {'var': float, 'es': float}
VaREngine.parametric_var(returns, conf) → {'var': float, 'es': float}
VaREngine.evt_var(returns, conf)        → {'var', 'es', 'gpd_shape', 'gpd_scale'}

# EVTAnalyzer
evt = EVTAnalyzer()
evt.fit(returns, confidence)
evt.hill_estimator(k_percentile=0.10) → {'tail_index', 'shape', 'threshold', 'k'}
evt.mean_excess_data() → pd.DataFrame with columns [threshold, mean_excess]

# VaRBacktest
bt = VaRBacktest()
bt.full_backtest(returns, var_series, confidence) → BacktestResult

# HMMRegimeDetector
hmm = HMMRegimeDetector(n_regimes)
hmm.fit(vol_series) → RegimeResult with .labels, .transition_matrix, .regime_means, .regime_stds, .current_regime

# ScenarioGenerator
gen = ScenarioGenerator.from_data(returns)
gen.generate(current, horizon, n_paths, seed=42) → dict with 'median', 'p5', 'p95' (pd.Series)
gen.stress_test(current, shock_multipliers=[1,1.5,2,3], horizon=60, n_paths=5000) → dict

# MarketGauge
gauge = MarketGauge()
gauge.assess(spread=spread, returns=returns) → {'composite': float, 'status': (en, zh), 'indicator_scores': {k: {'score': v}}}

# KalmanSignalExtractor
kalman = KalmanSignalExtractor()
kalman.fit(spread_series)  # NOTE: takes spread, NOT returns
kalman.result  # check .result not .is_fitted

# JSON serialization: ALL numpy/pandas → float(), int(), str()
```

---

## 6. 运行与验证

```bash
# 后端测试
cd /Users/liulu/Code/CNLocalGovSpread
python3.13 -m pytest tests/ -v   # 53 passed in 2.47s

# 启动后端 (mock data)
CLS_DATA__SOURCE=mock python3.13 -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload

# 启动前端
cd /Users/liulu/Code/CNLocalGovSpread/frontend
npm run dev   # http://localhost:3000

# TypeScript 检查
cd frontend && npx tsc --noEmit   # 0 errors

# Production build
cd frontend && npx next build     # succeeds
```

### 已验证的 API 响应

```
GET  /health            → status:ok, source:mock
GET  /data/statistics   → 4 columns (spread_all/5y/10y/30y), 含 ADF 检验
GET  /models/tournament → 5 models, winner_aic=FIGARCH
GET  /models/garch/detail → 1499 vol points + residuals + diagnostics
GET  /models/figarch    → d=0.188
POST /models/fit-custom → EGARCH converged
GET  /risk/evt          → tail_index=1.53, xi=-0.30
GET  /risk/backtest     → 101 violations (mock data)
GET  /regimes/hmm       → 3 regimes, current=0
POST /scenarios/stress  → 3 scenarios returned
GET  /analysis/sensitivity → base=1M, 3 variables
GET  /market/gauge      → composite=59.48, status=caution/警戒
```

---

## 7. 设计系统参考

### 色彩 (oklch in globals.css)
- Background: `oklch(0.12 0.01 260)` — near-black with blue tint
- Card: `oklch(0.17 0.012 260)`
- Primary: `oklch(0.7 0.15 250)` — saturated blue
- Chart palette: 5 colors at lightness ~0.7, chroma 0.15, hues 250/170/50/330/90

### 叙事结构 (每页必须遵循)
```
① WHY — 研究动机
② HOW — 方法论 (含 KaTeX 公式)
③ WHAT — 结果 (图表 + 读图指南)
④ SO WHAT — 诊断与解读
⑤ NOW WHAT — 投资含义
```

### 组件使用示例
```tsx
import { Section } from "@/components/narrative/section";
import { ProseBlock } from "@/components/narrative/prose-block";
import { Formula } from "@/components/narrative/formula";
import { ReadGuide } from "@/components/narrative/read-guide";
import { ParamTooltip } from "@/components/narrative/param-tooltip";
import { InsightCard } from "@/components/narrative/insight-card";
import { MetricCard } from "@/components/narrative/metric-card";
import { PlotlyChart, chartColors } from "@/components/charts/plotly-chart";
import { useTournament, useModelDetail, useRiskMetrics } from "@/hooks/use-api";

// 5-segment page structure:
<Section index={0} title="为什么需要条件波动率模型？">
  <ProseBlock><p>利差的波动不是恒定的...</p></ProseBlock>
</Section>

<Section index={1} title="方法论">
  <Formula block math={String.raw`\sigma^2_t = \omega + \alpha \varepsilon^2_{t-1} + \beta \sigma^2_{t-1}`} />
</Section>

<Section index={2} title="结果">
  <PlotlyChart data={[{x, y, type: 'scatter'}]} ariaLabel="条件波动率" />
  <ReadGuide>
    <p>蓝色线 = GARCH(1,1) 估计的每日条件波动率</p>
  </ReadGuide>
  <ParamTooltip name="α+β" value={0.96}
    tooltip={<>持续性参数，冲击半衰期 ≈ 17 天</>} />
</Section>
```

---

## 8. 下一个 Session 的任务: Phase 3–4

### Phase 3: Overview + Volatility 页

**`/analysis/overview`** — 利差全景
- 5 段叙事: WHY(为什么分析地方债利差) → HOW(数据概览) → WHAT(3个图表) → SO WHAT(分布特征) → NOW WHAT(链接到波动率)
- 图表: (1) 利差时序图 4条线+事件标注 (2) KDE分布+正态参考 (3) 期限结构散点
- 交互: 统计摘要表可排序, 单位根检验结果
- 用 `useDataStatistics()` + `useDataSummary()`

**`/analysis/volatility`** — 波动率建模
- 5 段叙事: WHY(条件波动率必要性) → HOW(KaTeX公式×4模型) → WHAT(tournament表+波动率对比图) → SO WHAT(残差诊断4-panel) → NOW WHAT(推荐模型+含义)
- 图表: (1) 条件波动率多模型叠加 (2) Tournament表 (3) QQ-plot+ACF+ARCH-LM+残差时序
- 交互: 模型选择器, 分布选择, "重新拟合"按钮
- 用 `useTournament()` + `useModelDetail(name)` + `useFigarch()` + `useFitCustom()`

### Phase 4: Risk + Regimes + Scenarios 页

**`/analysis/risk`** — 风险度量
- VaR对比柱状图, Hill plot, Mean excess plot, VaR backtest时序
- Confidence level slider, 用 `useRiskMetrics()` + `useEvt()` + `useBacktest()`

**`/analysis/regimes`** — 市场状态
- HMM状态时序(背景色带), 转移矩阵热力图, MarketGauge仪表盘
- 用 `useHmm()` + `useMarketGauge()`

**`/analysis/scenarios`** — 情景分析
- Fan chart(扇形图), 路径样本(50条半透明线), 压力测试表
- Horizon/N-paths slider, 用 `useScenarios()` + `useStress()`

---

## 9. 已知问题与注意事项

1. **`api/app.py`** 的 version 显示为 `src.core.config`（小 bug，可修但不紧急）
2. **Mock 数据** 模式下 backtest 有 101 violations（因为 mock 数据分布不理想），真实 CSV 数据应该更好
3. **EVT tail_index=1.53** 意味着分布较厚尾（正态是 ∞），ξ=-0.30 意味着 GPD 有上界（bounded tail）
4. **`dashboard/`** 目录仍存在，Phase 5 移入 `legacy/v3.0/`
5. **`components.json`** 在 frontend 根目录，shadcn v4 配置文件
6. **Tailwind v4** 使用 CSS-first 配置（`@theme inline` in globals.css），不是 tailwind.config.ts
7. **base-ui Sheet** 的 `open`/`onOpenChange` 控制模式与 Radix 不同，注意 API
8. 前端的 `next.config.ts` 通过 `rewrites` 将 `/api/*` 代理到 `localhost:8000`

---

## 10. 环境

- macOS, Node v22.19.0, npm 11.10.0, Python 3.13
- Timezone: Asia/Hong_Kong
- Date: 2026-08-01
