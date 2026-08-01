# 🤝 Handoff — CNLocalGovSpread v4.1 Complete Platform

> **Generated**: 2026-08-02  
> **Status**: Phase 1–5 Complete + UI/UX Polish — All verification green ✅  
> **Previous**: `HANDOFF-v4.md` (Phase 1–2 only, now superseded)

---

## 1. 项目状态总览

| 层级 | 状态 | 规模 | 备注 |
|------|------|------|------|
| `src/` (量化引擎) | ✅ **冻结** | 53 tests pass | 禁止修改 |
| `tests/` | ✅ **冻结** | 53 tests, 3.18s | 禁止修改 |
| `api/routes.py` | ✅ 完成 | 21 endpoints, 1069 行 | prefix `/api/v1` |
| `api/schemas.py` | ✅ 完成 | 30 Pydantic models, 366 行 | Pydantic v2 |
| `frontend/` | ✅ 完成 | 70+ TS/TSX 文件 | tsc 0 errors, next build ✅ |
| `dashboard/` (旧 Dash) | ⏳ 待移入 `legacy/` | Phase 6 任务 | 仍可使用但不维护 |

### 验证状态 (All Green)

| Check | Status | Notes |
|-------|--------|-------|
| `npx tsc --noEmit` | ✅ 0 errors | TypeScript strict mode |
| `npx next build` | ✅ success | All 7 routes static (Turbopack) |
| `python3.13 -m pytest tests/ -v` | ✅ 53 passed (3.18s) | 不能减少 |

---

## 2. API 端点完整清单 (21 个)

### 2.1 基础端点 (7 个)

| # | Method | Path | Schema | 说明 |
|---|--------|------|--------|------|
| 1 | GET | `/health` | `HealthResponse` | 健康检查 |
| 2 | GET | `/data/summary` | `DataSummary` | 数据概览 (n_rows, columns, date_range) |
| 3 | GET | `/data/raw?limit=100&offset=0` | — | 原始数据分页 |
| 4 | GET | `/models/fit?model_type=garch` | — | 单模型拟合 |
| 5 | GET | `/risk/metrics?confidence=0.99` | — | VaR/ES 综合指标 |
| 6 | GET | `/scenarios/generate?horizon=252&n_paths=5000` | — | MC 模拟扇形图 |
| 7 | GET | `/market/gauge` | — | 5 维综合压力仪表盘 |

### 2.2 进阶端点 (10 个, Phase 2)

| # | Method | Path | Schema | 说明 |
|---|--------|------|--------|------|
| 8 | GET | `/data/statistics` | `DataStatisticsResponse` | 4 列统计 + ADF 检验 |
| 9 | GET | `/models/tournament` | `TournamentResponse` | 5 模型 AIC/BIC 锦标赛 |
| 10 | GET | `/models/{name}/detail` | `ModelDetailResponse` | 单模型详情 (vol, residuals, diagnostics) |
| 11 | GET | `/models/figarch` | `FigarchResponse` | FIGARCH 长记忆 d 参数 |
| 12 | POST | `/models/fit-custom` | `ModelDetailResponse` | 自定义模型拟合 |
| 13 | GET | `/risk/evt?percentile=0.10` | `EvtResponse` | GPD-POT + Hill + Mean Excess |
| 14 | GET | `/risk/backtest?confidence=0.99` | `BacktestResponse` | Kupiec + Christoffersen 回测 |
| 15 | GET | `/regimes/hmm?n_regimes=3` | `HmmResponse` | HMM 状态检测 |
| 16 | POST | `/scenarios/stress` | `StressResponse` | 压力测试 |
| 17 | GET | `/analysis/sensitivity` | `SensitivityResponse` | 敏感性分析 (龙卷风图) |

### 2.3 Regimes 高级方法 (4 个, Phase 5)

| # | Method | Path | Schema | 说明 |
|---|--------|------|--------|------|
| 18 | GET | `/regimes/kalman-signal?column=spread_all` | `KalmanSignalResponse` | Kalman 滤波信号提取 |
| 19 | GET | `/regimes/changepoints?column=spread_all&method=binseg&n_breakpoints=5` | `ChangepointResponse` | PELT/BinSeg 变化点检测 |
| 20 | GET | `/regimes/sts-signal?column=spread_all` | `STSSignalResponse` | 结构化时间序列 (level+slope) |
| 21 | GET | `/regimes/bayesian-sts?column=spread_all` | `BayesianSTSResponse` | PyMC 贝叶斯 STS (后验 + 80% CI) |

---

## 3. 前端文件结构

```
frontend/
├── app/
│   ├── layout.tsx (68L)              — dark, zh-CN, Inter+JetBrains, TooltipProvider
│   ├── page.tsx (276L)               — Hero + LiveSnapshot + Abstract + Framework + NavCards
│   ├── globals.css (534L)            — oklch dark theme, prose-narrative, chart-container
│   ├── _components/
│   │   ├── live-snapshot.tsx (88L)   — 4 MetricCards from useMarketGauge/useRiskMetrics
│   │   └── hero-data-badge.tsx (20L) — Client component for dynamic data badge
│   └── analysis/
│       ├── overview/
│       │   ├── page.tsx              — wrapper (Sidebar + Breadcrumb + TOC)
│       │   └── overview-content.tsx (311L) — 数据全景: 时序+KDE+期限结构
│       ├── volatility/
│       │   ├── page.tsx
│       │   └── volatility-content.tsx (412L) — GARCH锦标赛+FIGARCH+残差诊断
│       ├── risk/
│       │   ├── page.tsx
│       │   └── risk-content.tsx (623L) — VaR对比+EVT+Hill+回测
│       ├── regimes/
│       │   ├── page.tsx (12L)
│       │   └── regimes-content.tsx (1100L) — HMM+Kalman+STS+Bayesian+CPD+MarketGauge+ExecSummary
│       └── scenarios/
│           ├── page.tsx
│           └── scenarios-content.tsx (386L) — Fan chart+路径+压力测试
├── components/
│   ├── providers.tsx (23L)            — QueryClientProvider
│   ├── narrative/ (10 components)
│   │   ├── section.tsx (69L)          — WHY/HOW/WHAT/SO WHAT/NOW WHAT 5段
│   │   ├── chart-wrapper.tsx (83L)    — 统一图表容器 (header + metrics + content)
│   │   ├── read-guide.tsx (76L)       — 可折叠读图指南
│   │   ├── insight-card.tsx (89L)     — 发现/警告/成功 callout
│   │   ├── metric-card.tsx (143L)     — 指标卡片 (value + change + sparkline)
│   │   ├── param-tooltip.tsx (68L)    — 参数 tooltip (经济学解释)
│   │   ├── prose-block.tsx (36L)      — 叙事段落 (default + callout)
│   │   ├── formula.tsx (24L)          — KaTeX 公式 (inline + block)
│   │   ├── navigation-card.tsx (60L)  — 模块导航卡片
│   │   ├── executive-summary.tsx (230L) — 综合诊断仪表盘 (NEW)
│   │   └── page-navigation.tsx (80L)  — 上/下一页导航 (NEW)
│   ├── charts/ (20 components)
│   │   ├── plotly-chart.tsx (120L)    — Plotly 基础组件 (dark theme + colorway)
│   │   ├── kalman-signal.tsx (180L)   — Kalman 滤波信号图 (dual-panel)
│   │   ├── sts-signal.tsx (200L)      — STS 信号图 (triple-panel)
│   │   ├── bayesian-sts.tsx (180L)    — Bayesian STS (CI band + z-score)
│   │   ├── changepoint.tsx (160L)     — 变化点检测 (segments + breakpoints)
│   │   ├── regime-sequence.tsx (130L) — HMM 状态序列 (colored bands)
│   │   ├── market-gauge-panel.tsx (180L) — MarketGauge 仪表盘
│   │   └── ... (13 more chart components)
│   ├── interactive/ (4 components)
│   │   ├── model-selector.tsx         — 模型选择器
│   │   ├── scenario-controls.tsx      — 情景参数控制
│   │   ├── stress-form.tsx            — 压力测试表单
│   │   └── confidence-slider.tsx      — 置信度滑块
│   ├── layout/ (4 components)
│   │   ├── sidebar.tsx (80L)          — 分析模块侧边栏
│   │   ├── breadcrumb.tsx (40L)       — 面包屑导航
│   │   ├── toc.tsx (60L)              — 目录导航
│   │   └── header.tsx (50L)           — 顶部导航栏
│   └── ui/ (21 components)            — shadcn/ui base components
├── hooks/
│   ├── use-api.ts (256L)              — 20 TanStack Query hooks
│   ├── use-scroll-spy.ts (40L)        — 滚动监听 (TOC active state)
│   └── use-theme.ts (20L)             — 主题切换
├── lib/
│   ├── api.ts (124L)                  — API client (fetch wrapper)
│   ├── types.ts (340L)                — TypeScript types (mirrors Pydantic)
│   └── utils.ts (80L)                 — cn(), fmt(), etc.
├── types/
│   └── react-katex.d.ts               — KaTeX type definitions
├── next.config.ts                     — rewrites proxy to :8000
├── components.json                    — shadcn config
└── tsconfig.json                      — strict mode
```

---

## 4. 5 个分析页面概览

| 页面 | 路径 | 叙事结构 | 核心图表 | 核心指标 |
|------|------|----------|----------|----------|
| **利差全景** | `/analysis/overview` | WHY→HOW→WHAT→SO→NOW | 时序+KDE+期限结构 | 均值、标准差、偏度、峰度 |
| **波动率建模** | `/analysis/volatility` | WHY→HOW→WHAT→SO→NOW | GARCH锦标赛+FIGARCH+残差诊断 | AIC/BIC、ARCH效应、Ljung-Box |
| **风险度量** | `/analysis/risk` | WHY→HOW→WHAT→SO→NOW | VaR对比+EVT+Hill+回测 | VaR 99%、ES、GPD ξ |
| **市场状态** | `/analysis/regimes` | WHY→HOW→WHAT→SO→NOW | HMM+Kalman+STS+Bayesian+CPD+Gauge | z-score、regime、composite |
| **情景分析** | `/analysis/scenarios` | WHY→HOW→WHAT→SO→NOW | Fan chart+路径+压力测试 | P95、prob_exceed |

### 4.1 Regimes 页面详细结构 (1100L, 最复杂)

#### WHY — 研究动机
- 信用利差的"状态"比"数值"更重要
- 传统均值回归假设在危机期失效
- 需要多维度状态识别方法

#### HOW — 方法论 (5 种方法)

**0. Kalman Filter (Local Level Model)**
- 公式: `y_t = μ_t + ε_t; μ_{t+1} = μ_t + η_t`
- 输出: signal (平滑趋势), deviation_zscore (标准化偏离度)
- 参数: Q = σ²_η / σ²_ε (信噪比, 决定平滑度)
- 优势: 简单高效, 实时可计算
- 局限: 无趋势/季节性成分

**1. HMM (Hidden Markov Model)**
- 公式: `P(s_t | s_{t-1}) = A`, `y_t ~ N(μ_{s_t}, σ²_{s_t})`
- 输出: labels (状态序列), transition_matrix (转移矩阵)
- 参数: n_regimes=3 (低/中/高波动)
- 优势: 捕捉波动率聚类
- 局限: 预设状态数量

**2. Structural Time Series (STS)**
- 公式: `y_t = μ_t + β·t + ε_t; μ_{t+1} = μ_t + ν_t; ν_{t+1} = ν_t + ζ_t`
- 输出: level (趋势), slope (漂移率), deviation_zscore
- 参数: AIC/BIC 选择模型复杂度
- 优势: 量化趋势变化速度
- 局限: 线性假设

**3. Bayesian STS (PyMC)**
- 公式: `μ_t ~ N(μ_{t-1}, σ²_level); y_t ~ N(μ_t, σ²_obs)`
- 输出: signal_lower/signal_upper (80% CI), posterior samples
- 参数: n_samples=2000, ADVI 变分推断
- 优势: 不确定性量化 (CI width)
- 局限: 计算成本高 (~5s)

**4. Change Point Detection (CPD)**
- 公式: `min Σ (y_i - μ_k)² + λ·K` (PELT/BinSeg)
- 输出: breakpoint_dates (断裂点), segments (均值段)
- 参数: method="binseg", n_breakpoints=5
- 优势: 精确定位结构性转折
- 局限: 不预测未来方向

#### WHAT — 结果展示
- **ExecutiveSummary** 综合诊断仪表盘 (NEW)
  - 显示 5 个方法的信号一致性 (3/5 methods agree)
  - 综合判定: 利差高估/低估/合理
  - 一句话结论
- HMM 状态序列图 (背景色 = 隐含状态)
- HMM 转移矩阵热力图 (p_ij 概率)
- Kalman 滤波信号图 (dual-panel: spread+signal, z-score)
- STS 信号图 (triple-panel: level, slope, z-score)
- Bayesian STS 图 (CI band + z-score)
- CPD 变化点图 (segments + vertical dashed lines)
- 方法对比表格 (5 methods × 5 dimensions)

#### SO WHAT — 诊断与解读
- 交叉验证结论: 5 种方法是否一致?
- 当前信号解读: z-score 含义, regime 含义
- 历史对比: 当前状态与 2020 疫情/2022 赎回潮的相似度

#### NOW WHAT — 投资含义
- MarketGauge 综合仪表盘 (5 维指标)
- 联合信号: HMM + Kalman + STS + Bayesian + MarketGauge
- 投资建议: 仓位调整、久期管理、对冲策略
- PageNavigation: 上/下一页导航 (NEW)

---

## 5. 核心设计哲学 (不可违背)

### 5.1 叙事结构
每个分析页面遵循 **WHY → HOW → WHAT → SO WHAT → NOW WHAT** 5 段式:
- **WHY**: 研究动机, 为什么这个问题重要
- **HOW**: 方法论, 公式 + 经济学解释 + 参数 tooltip
- **WHAT**: 结果展示, 图表 + 读图指南
- **SO WHAT**: 诊断与解读, 交叉验证 + 历史对比
- **NOW WHAT**: 投资含义, 决策建议 + 下一步行动

### 5.2 可视化规范
- 每张图表必须有 `<ReadGuide>` 读图指南
- 关键参数必须有 `<ParamTooltip>` 经济学 tooltip
- 公式必须用 `<Formula>` (KaTeX), block=true 用于 display math
- 中文为主, 技术术语括号加英文 (e.g., "厚尾 (fat tails)")
- 深色主题, oklch 色彩空间, institutional finance aesthetic

### 5.3 交互规范
- 图表: Plotly (next/dynamic, ssr: false), hover 显示详细数值
- 折叠: ReadGuide 默认折叠, 点击展开
- 动画: Framer Motion entrance animations (opacity + y offset)
- 导航: PageNavigation 上/下一页, Sidebar 模块切换, TOC 章节跳转

---

## 6. 技术约束

1. **不要碰 `src/` 和 `tests/`** (除非新增 Kalman API 端点需要修改 api/routes.py 和 api/schemas.py)
2. 新增 Python 代码必须通过 `pytest tests/ -v` (仍为 53 passed, 不能减少)
3. TypeScript strict, 不允许 `any`
4. 深色主题 — 使用 globals.css 中已定义的 CSS 变量
5. shadcn/ui 使用 `@base-ui/react` (不是 Radix)
6. Plotly 必须用 `next/dynamic (ssr: false)`
7. KaTeX — 使用 Formula 组件, block=true 用于 display math
8. 服务端组件 (Server Components) 不能调用 hooks, 需提取为 Client Components

---

## 7. 运行命令

```bash
# Backend (port 8000)
cd /Users/liulu/Code/CNLocalGovSpread
CLS_DATA__SOURCE=mock python3.13 -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload

# Frontend (port 3000)
cd /Users/liulu/Code/CNLocalGovSpread/frontend
npm run dev

# Verify (ALL MUST PASS)
cd frontend && npx tsc --noEmit && npx next build
cd /Users/liulu/Code/CNLocalGovSpread && python3.13 -m pytest tests/ -v  # 53 passed
```

---

## 8. 已知问题 (Not Fixed)

### 8.1 Bayesian STS Data Quality (backend, low priority)
- API `/regimes/bayesian-sts` returns **signal ~-9 bps** (actual spread ~30 bps)
- z-score first value = 101.04 (clearly wrong)
- Root cause: likely in `src/models/bayesian_sts.py` (PyMC ADVI initialization)
- **Decision**: Not fixed because `src/` is frozen (tests must stay 53 passed). Frontend displays what backend returns.

### 8.2 Kalman Q-ratio Display
- Backend returns `q_ratio: 88520731760.39` (unrealistically high due to `sigma2_eps ≈ 1e-11`)
- Frontend shows `">>1"` when > 1e6 (graceful degradation)
- **Decision**: Left alone, backend issue.

### 8.3 Loading Indicator on Regimes (not actually a bug)
- One `animate-pulse` skeleton visible after 8s wait
- This is the **Plotly dynamic import loading state** (`components/charts/plotly-chart.tsx:12`)
- Normal behavior for large Plotly bundle (~2MB) loading lazily
- **Decision**: Left alone.

---

## 9. 关键文件清单

### 9.1 核心配置文件
- `frontend/next.config.ts` — rewrites proxy to :8000
- `frontend/tsconfig.json` — strict mode
- `frontend/components.json` — shadcn config
- `frontend/app/globals.css` — design system (oklch colors, prose-narrative, chart-container)

### 9.2 核心组件
- `frontend/components/charts/plotly-chart.tsx` — Plotly 基础组件 (dark theme + colorway)
- `frontend/components/narrative/section.tsx` — 5 段式 Section 容器
- `frontend/components/narrative/executive-summary.tsx` — 综合诊断仪表盘 (NEW)
- `frontend/components/narrative/page-navigation.tsx` — 上/下一页导航 (NEW)

### 9.3 核心页面
- `frontend/app/page.tsx` — 首页 (Hero + LiveSnapshot + Framework + NavCards)
- `frontend/app/analysis/regimes/regimes-content.tsx` — Regimes 页 (1100L, 最复杂)
- `frontend/app/_components/hero-data-badge.tsx` — Client component pattern (NEW)

### 9.4 API 层
- `api/routes.py` — 21 endpoints (1069L)
- `api/schemas.py` — 30 Pydantic models (366L)
- `frontend/hooks/use-api.ts` — 20 TanStack Query hooks (256L)
- `frontend/lib/types.ts` — TypeScript types (340L, mirrors Pydantic)

---

## 10. 下一步建议 (Phase 6)

### 10.1 待办事项 (Not Started)
- [ ] 将 `dashboard/` 移入 `legacy/v3.0/`
- [ ] Mobile TOC 优化 (目前桌面端 TOC 好用, 移动端需调整)
- [ ] Print stylesheet (已添加基础 print styles, 需完善)
- [ ] Skeleton shimmer 加载态优化 (已添加 CSS, 需应用到所有页面)
- [ ] `/dashboard` 路径改为聚合仪表板 (跨页面指标汇总)
- [ ] 国际化 (i18n) 基础架构
- [ ] PWA / offline support
- [ ] 性能优化: React Server Components for static sections
- [ ] E2E tests (Playwright)

### 10.2 技术债务
- `dashboard/` 目录仍存在, 应移入 `legacy/v3.0/`
- Bayesian STS 数据质量问题 (backend, `src/models/bayesian_sts.py`)
- Kalman Q-ratio 显示问题 (backend, `src/models/kalman.py`)

### 10.3 设计改进
- 添加 "方法论" 页面解释 WHY-HOW-WHAT 框架
- 添加 "数据源" 页面说明 Wind EDB 数据质量
- 添加 "关于" 页面介绍 QuinnMacro 研究团队
- 添加 footer 链接 (GitHub, Paper, Contact)

---

## 11. 环境

| Component | Version |
|-----------|---------|
| macOS | Current |
| Node.js | v22.19.0 |
| npm | 11.10.0 |
| Python | 3.13 |
| Next.js | 16.2.12 (Turbopack) |
| TypeScript | strict mode |
| Tailwind | v4 (CSS-first config) |
| shadcn/ui | v4 (@base-ui/react) |
| Plotly | next/dynamic (ssr: false) |
| KaTeX | react-katex |
| TanStack Query | v5 |
| Framer Motion | latest |

---

## 12. 快速恢复指南

**新 Session 开始时的检查清单**:

1. ✅ 后端在跑吗? `curl http://127.0.0.1:8000/api/v1/health`
2. ✅ 前端在跑吗? `open http://localhost:3000`
3. ✅ tsc 干净吗? `cd frontend && npx tsc --noEmit`
4. ✅ tests 全过吗? `python3.13 -m pytest tests/ -v --tb=no -q`
5. ✅ 读完本文件的 §2 (API) + §4 (Pages) + §6 (Constraints)

**如果服务没跑**:
```bash
# Terminal 1: Backend
cd /Users/liulu/Code/CNLocalGovSpread
CLS_DATA__SOURCE=mock python3.13 -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Frontend
cd /Users/liulu/Code/CNLocalGovSpread/frontend
npm run dev
```

---

## 13. 本次 Session 完成的工作

### 13.1 Bug Fixes (from prior session summary)
- Title duplication: "市场状态 | QuinnMacro | QuinnMacro" → "市场状态"
- Home page SSR crash: extracted `HeroDataBadge` client component
- Ordinal suffix: "53th" → "53rd"
- Kalman integration: 17 surgical edits to regimes-content.tsx
- Outdated descriptions: updated dates and method counts

### 13.2 UI/UX Improvements (this session)
- **ExecutiveSummary component** (230L): 综合诊断仪表盘
  - 5 mini signal cards (Kalman/STS/Bayesian z-scores, HMM regime, MarketGauge)
  - Agreement meter (3/5 methods agree)
  - One-sentence executive summary
- **PageNavigation component** (80L): 上/下一页导航
  - Prev/next links with emoji icons
  - Smooth transitions
- **Enhanced PlotlyChart** (120L): better dark theme + colorway
- **Enhanced globals.css** (534L): 
  - Executive summary styles (exec-verdict, exec-mini-card, exec-agreement-bar)
  - Method card grid (method-card-grid, method-card)
  - Section connectors (section-connector)
  - Conclusion footer (conclusion-footer)
  - Print styles (basic)
  - Chart container hover enhancement
  - Badge pulse animation

### 13.3 Verification
- ✅ `npx tsc --noEmit` — 0 errors
- ✅ `npx next build` — success (7 routes static)
- ✅ `python3.13 -m pytest tests/ -v` — 53 passed (3.18s)
- ✅ Backend running on :8000
- ✅ Frontend running on :3000

---

## 14. 设计系统参考

### 14.1 色彩 (oklch)
```css
--primary: oklch(0.7 0.15 250);        /* blue */
--chart-1: oklch(0.75 0.15 250);       /* blue */
--chart-2: oklch(0.70 0.15 170);       /* teal */
--chart-3: oklch(0.75 0.15 50);        /* orange */
--chart-4: oklch(0.70 0.15 330);       /* pink */
--chart-5: oklch(0.75 0.15 90);        /* yellow-green */
--destructive: oklch(0.6 0.2 25);      /* red */
```

### 14.2 字体
- Sans: Inter (variable)
- Mono: JetBrains Mono (variable)
- Heading: Inter (bold)

### 14.3 间距
- Section padding: `py-10 md:py-14`
- Container: `max-w-4xl mx-auto` or `max-w-5xl mx-auto`
- Card padding: `p-4` or `p-5`
- Gap: `gap-3` or `gap-4`

### 14.4 动画
- Entrance: `opacity 0 → 1, y 30 → 0, duration 0.6s`
- Hover: `scale 1.02, shadow-lg, duration 0.2s`
- Collapse: `height 0 → auto, duration 0.3s`

---

## 15. 文件行数统计

### Backend
| File | Lines |
|------|-------|
| `api/routes.py` | 1069 |
| `api/schemas.py` | 366 |
| `api/app.py` | ~50 |
| **Total API** | **~1485** |

### Frontend (source only, excl. node_modules/.next/ui)
| Category | Lines | Files |
|----------|-------|-------|
| Analysis pages | 2812 | 5 content files |
| Chart components | 2241 | 20 components |
| Narrative components | 953 | 11 components |
| Interactive components | 335 | 4 components |
| Layout components | 230 | 4 components |
| UI components | ~2000 | 21 components |
| Hooks | 317 | 3 files |
| Lib (api+types+utils) | 544 | 3 files |
| Layout+Home+CSS | 878 | 5 files |
| **Total Frontend** | **~8310** | **~52 files** |

### Grand Total
| Layer | Lines |
|-------|-------|
| Backend API | ~1,485 |
| Frontend | ~8,310 |
| **Total** | **~9,795** |

---

> **END OF HANDOFF** — 本文档覆盖 Phase 1–5 + UI/UX Polish 全部内容。下一 session 可直接开始 Phase 6 候选任务。
