# CNLocalGovSpread Frontend — Agent Instructions

<!-- BEGIN:nextjs-agent-rules -->
# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.
<!-- END:nextjs-agent-rules -->

## Architecture

- **Framework**: Next.js 16 with App Router
- **Language**: TypeScript strict (no `any`)
- **Styling**: Tailwind CSS v4 (CSS-first config via `@theme inline`)
- **UI Library**: shadcn/ui v4 (`@base-ui/react`, NOT Radix)
- **Charts**: Plotly (dynamic import, ssr: false) + dark theme
- **Math**: KaTeX via react-katex
- **Data**: TanStack Query v5 → FastAPI backend at `localhost:8000`

## File Organization

| Directory | Purpose |
|-----------|---------|
| `app/analysis/*` | 5 analysis pages (each has `page.tsx` + `*-content.tsx`) |
| `components/charts/` | 20 Plotly chart components |
| `components/narrative/` | 9 narrative structure components |
| `components/interactive/` | 4 user interaction components |
| `hooks/` | TanStack Query hooks + scroll spy |
| `lib/` | API client + types + utilities |

## Critical Rules

1. **Do NOT modify `../src/` or `../tests/`** — 53 tests must stay passing
2. **Always use `next/dynamic`** for Plotly (SSR incompatible)
3. **Use CSS variables** from `globals.css` — never hardcode hex/rgb colors
4. **Chinese text primary** with English technical terms in parentheses
5. **Every chart** needs `<ChartWrapper>` + `<ReadGuide>` (读图指南)
6. **Every formula** needs `<Formula>` component (KaTeX)
7. **Every key parameter** needs `<ParamTooltip>` with economic explanation
8. **Every page** follows: WHY → HOW → WHAT → SO WHAT → NOW WHAT

## Key Imports

```typescript
// Narrative
import { Section } from "@/components/narrative/section";
import { ProseBlock } from "@/components/narrative/prose-block";
import { Formula } from "@/components/narrative/formula";
import { ReadGuide } from "@/components/narrative/read-guide";
import { ParamTooltip } from "@/components/narrative/param-tooltip";
import { InsightCard } from "@/components/narrative/insight-card";
import { MetricCard } from "@/components/narrative/metric-card";
import { ChartWrapper } from "@/components/narrative/chart-wrapper";

// Charts
import { PlotlyChart, chartColors } from "@/components/charts/plotly-chart";

// Data
import { useKalmanSignal, useStsSignal, useBayesianSts } from "@/hooks/use-api";

// Utilities
import { cn, fmt } from "@/lib/utils";
```

## Verification

```bash
npx tsc --noEmit     # 0 errors
npx next build        # success
```
