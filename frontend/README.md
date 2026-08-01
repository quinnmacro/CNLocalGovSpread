# CNLocalGovSpread Frontend — Next.js Analytical Platform

> Bloomberg Terminal × Linear aesthetic · Dark theme · Chinese-primary UI

---

## Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| Next.js | 16 | App Router, React Server Components |
| TypeScript | strict | Full type safety (no `any`) |
| Tailwind CSS | v4 | CSS-first config (`@theme inline` in globals.css) |
| shadcn/ui | v4 | UI primitives via `@base-ui/react` |
| Plotly | latest | Interactive charts (`next/dynamic`, ssr: false) |
| KaTeX | react-katex | Math formula rendering |
| TanStack Query | v5 | Data fetching + caching (20 hooks) |
| Framer Motion | latest | Animations + transitions |

## Getting Started

```bash
# Install dependencies
npm install

# Start development server
npm run dev  # → http://localhost:3000

# Type checking (must pass)
npx tsc --noEmit  # 0 errors required

# Production build (must succeed)
npx next build
```

**Backend required**: The frontend proxies `/api/*` to `localhost:8000` via `next.config.ts` rewrites.

```bash
# Start backend (from project root)
cd ..
CLS_DATA__SOURCE=mock python3.13 -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx              — Root layout (dark, zh-CN, fonts, TooltipProvider)
│   ├── page.tsx                — Home: Hero + LiveSnapshot + Framework + NavCards
│   ├── globals.css             — Design system (oklch colors, typography, animations)
│   ├── _components/
│   │   └── live-snapshot.tsx   — Home page metric cards
│   └── analysis/
│       ├── overview/           — 利差全景 (time series, KDE, term structure)
│       ├── volatility/         — 波动率建模 (GARCH tournament, FIGARCH, diagnostics)
│       ├── risk/               — 风险度量 (VaR, EVT, backtesting)
│       ├── regimes/            — 市场状态 (HMM+Kalman+STS+Bayesian+CPD+Gauge)
│       └── scenarios/          — 情景分析 (fan chart, MC paths, stress tests)
├── components/
│   ├── charts/ (20)            — Plotly chart components
│   ├── narrative/ (9)          — Section, Formula, ReadGuide, ParamTooltip, etc.
│   ├── interactive/ (4)        — Sliders, selectors, data tables
│   ├── layout/ (4)             — Navbar, Sidebar, Breadcrumb, Footer
│   └── ui/ (21)                — shadcn/ui primitives
├── hooks/
│   ├── use-api.ts              — 20 TanStack Query hooks for all 21 API endpoints
│   ├── use-scroll-spy.ts       — IntersectionObserver for TOC scroll spy
│   └── use-theme.ts            — Theme stub (dark only)
├── lib/
│   ├── api.ts                  — Fetch wrapper + typed convenience methods
│   ├── types.ts                — 38 TypeScript interfaces (mirror Pydantic schemas)
│   └── utils.ts                — cn() class merger + fmt() number formatter
├── types/
│   └── react-katex.d.ts        — KaTeX type declarations
├── next.config.ts              — Rewrites proxy (/api/* → localhost:8000)
├── components.json             — shadcn/ui v4 configuration
└── tsconfig.json               — TypeScript strict config
```

## Page Pattern

Every analysis page follows the **5-segment narrative structure**:

```
① WHY    — 研究动机 (why this analysis matters)
② HOW    — 方法论 (methodology + KaTeX formulas)
③ WHAT   — 结果 (charts + data + read guides)
④ SO WHAT — 诊断与解读 (diagnostics + insights)
⑤ NOW WHAT — 投资含义 (investment implications)
```

Each page is split into two files:
- `page.tsx` — wrapper (Sidebar + Breadcrumb + TOC + loading state)
- `*-content.tsx` — actual content (sections, charts, formulas)

## Component Catalog

### Narrative Components
| Component | File | Purpose |
|-----------|------|---------|
| `Section` | `section.tsx` | WHY/HOW/WHAT segment wrapper with index + title |
| `ProseBlock` | `prose-block.tsx` | Body text with `prose-narrative` typography |
| `Formula` | `formula.tsx` | KaTeX math (block=true for display, false for inline) |
| `ReadGuide` | `read-guide.tsx` | Collapsible "📖 读图指南" for chart interpretation |
| `ParamTooltip` | `param-tooltip.tsx` | Hover tooltip for economic parameter explanations |
| `InsightCard` | `insight-card.tsx` | Callout card (variant: info/warning/success) |
| `MetricCard` | `metric-card.tsx` | Value display + trend icon + sparkline |
| `NavigationCard` | `navigation-card.tsx` | Hover-animated page entry card |
| `ChartWrapper` | `chart-wrapper.tsx` | Standardized chart container with title + border |

### Chart Components
| Component | File | Page |
|-----------|------|------|
| `PlotlyChart` | `plotly-chart.tsx` | Base (dark theme, dynamic import) |
| `TimeSeries` | `time-series.tsx` | Overview |
| `Distribution` | `distribution.tsx` | Overview |
| `TermStructure` | `term-structure.tsx` | Overview |
| `VolatilityOverlay` | `volatility-overlay.tsx` | Volatility |
| `TournamentTable` | `tournament-table.tsx` | Volatility |
| `ResidualDiagnostics` | `residual-diagnostics.tsx` | Volatility |
| `VarComparison` | `var-comparison.tsx` | Risk |
| `HillPlot` | `hill-plot.tsx` | Risk |
| `MeanExcessPlot` | `mean-excess-plot.tsx` | Risk |
| `VarBacktest` | `var-backtest.tsx` | Risk |
| `RegimeSequence` | `regime-sequence.tsx` | Regimes |
| `TransitionHeatmap` | `transition-heatmap.tsx` | Regimes |
| `MarketGaugePanel` | `market-gauge-panel.tsx` | Regimes |
| `KalmanSignal` | `kalman-signal.tsx` | Regimes |
| `Changepoint` | `changepoint.tsx` | Regimes |
| `StsSignal` | `sts-signal.tsx` | Regimes |
| `BayesianSts` | `bayesian-sts.tsx` | Regimes |
| `FanChart` | `fan-chart.tsx` | Scenarios |
| `StressTable` | `stress-table.tsx` | Scenarios |

## Design System

- **Colors**: `oklch()` throughout — see `globals.css` for full palette
- **Chart palette**: `chartColors` from `plotly-chart.tsx` (5 colors: blue/teal/amber/pink/green)
- **Number formatting**: `fmt()` from `@/lib/utils` (adaptive decimal places)
- **Language**: Chinese-primary, English technical terms in parentheses

## Constraints

1. TypeScript strict — no `any` allowed
2. Plotly must use `next/dynamic` with `ssr: false`
3. shadcn/ui uses `@base-ui/react` (not Radix)
4. All API responses fully typed via `lib/types.ts`
5. Every chart needs `ChartWrapper` + `ReadGuide`
6. Every formula needs `Formula` component (KaTeX)
7. Every key parameter needs `ParamTooltip` with economic explanation

## Further Documentation

- [`../HANDOFF-v4.1.md`](../HANDOFF-v4.1.md) — Complete project state reference
- [`../CHANGELOG.md`](../CHANGELOG.md) — Version history
- [`../README.md`](../README.md) — Project overview
