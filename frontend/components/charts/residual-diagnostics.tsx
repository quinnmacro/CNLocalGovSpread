"use client";
import { PlotlyChart, chartColors } from "./plotly-chart";
import type { TimePoint, DiagnosticsInfo } from "@/lib/types";
import type { Data } from "plotly.js";
import { fmt } from "@/lib/utils";
import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/card";

interface ResidualDiagnosticsProps {
  residuals: TimePoint[];
  diagnostics: DiagnosticsInfo | null;
  height?: number;
}

function computeQQ(data: number[]): { theoretical: number[]; sample: number[] } {
  const sorted = [...data].sort((a, b) => a - b);
  const n = sorted.length;
  const theoretical = sorted.map((_, i) => {
    const p = (i + 0.5) / n;
    return qnorm(p);
  });
  return { theoretical, sample: sorted };
}

// Approximate inverse normal CDF (Abramowitz & Stegun)
function qnorm(p: number): number {
  if (p <= 0) return -4;
  if (p >= 1) return 4;
  if (p === 0.5) return 0;
  const a = p < 0.5 ? p : 1 - p;
  const t = Math.sqrt(-2 * Math.log(a));
  const c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
  const d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
  const z = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);
  return p < 0.5 ? -z : z;
}

function computeACF(data: number[], maxLag: number): number[] {
  const n = data.length;
  const mean = data.reduce((a, b) => a + b, 0) / n;
  const variance = data.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  if (variance === 0) return Array(maxLag).fill(0);
  const acf: number[] = [];
  for (let lag = 0; lag <= maxLag; lag++) {
    let sum = 0;
    for (let i = 0; i < n - lag; i++) {
      sum += (data[i] - mean) * (data[i + lag] - mean);
    }
    acf.push(sum / (n * variance));
  }
  return acf;
}

export function ResidualDiagnostics({ residuals, diagnostics, height = 600 }: ResidualDiagnosticsProps) {
  const values = residuals.map(r => r.value);
  const { theoretical, sample } = computeQQ(values);
  const acfResid = computeACF(values, 20);
  const squaredValues = values.map(v => v * v);
  const acfSquared = computeACF(squaredValues, 20);
  const lags = Array.from({ length: 21 }, (_, i) => i);

  const panels: { title: string; traces: Data[]; layout?: Record<string, unknown> }[] = [
    {
      title: "QQ Plot",
      traces: [
        {
          x: theoretical, y: sample,
          type: "scatter" as const, mode: "markers" as const,
          marker: { color: chartColors[0], size: 3 },
          showlegend: false,
        },
        {
          x: [Math.min(...theoretical), Math.max(...theoretical)],
          y: [Math.min(...theoretical), Math.max(...theoretical)],
          type: "scatter" as const, mode: "lines" as const,
          line: { color: "oklch(0.5 0.1 25)", dash: "dash" as const, width: 1 },
          showlegend: false,
        },
      ],
      layout: { xaxis: { title: { text: "理论分位数" } }, yaxis: { title: { text: "样本分位数" } } },
    },
    {
      title: "ACF — 残差",
      traces: [{
        x: lags, y: acfResid,
        type: "bar" as const,
        marker: { color: chartColors[1] },
        showlegend: false,
      },
      {
        x: lags, y: lags.map(() => 1.96 / Math.sqrt(values.length)),
        type: "scatter" as const, mode: "lines" as const,
        line: { color: "oklch(0.5 0.1 25)", dash: "dash" as const, width: 1 },
        showlegend: false,
      },
      {
        x: lags, y: lags.map(() => -1.96 / Math.sqrt(values.length)),
        type: "scatter" as const, mode: "lines" as const,
        line: { color: "oklch(0.5 0.1 25)", dash: "dash" as const, width: 1 },
        showlegend: false,
      }],
      layout: { xaxis: { title: { text: "滞后阶数" } }, yaxis: { title: { text: "ACF" } } },
    },
    {
      title: "ACF — 残差²",
      traces: [{
        x: lags, y: acfSquared,
        type: "bar" as const,
        marker: { color: chartColors[2] },
        showlegend: false,
      },
      {
        x: lags, y: lags.map(() => 1.96 / Math.sqrt(values.length)),
        type: "scatter" as const, mode: "lines" as const,
        line: { color: "oklch(0.5 0.1 25)", dash: "dash" as const, width: 1 },
        showlegend: false,
      },
      {
        x: lags, y: lags.map(() => -1.96 / Math.sqrt(values.length)),
        type: "scatter" as const, mode: "lines" as const,
        line: { color: "oklch(0.5 0.1 25)", dash: "dash" as const, width: 1 },
        showlegend: false,
      }],
      layout: { xaxis: { title: { text: "滞后阶数" } }, yaxis: { title: { text: "ACF" } } },
    },
    {
      title: "标准化残差",
      traces: [{
        x: residuals.map(r => r.date),
        y: values,
        type: "scatter" as const, mode: "lines" as const,
        line: { color: chartColors[3], width: 1 },
        showlegend: false,
      }],
      layout: { xaxis: { title: { text: "日期" } }, yaxis: { title: { text: "标准化残差" } } },
    },
  ];

  const halfHeight = height / 2;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {panels.map((panel, i) => (
          <Card key={i} className="p-3">
            <h4 className="text-sm font-medium text-foreground mb-2">{panel.title}</h4>
            <PlotlyChart
              data={panel.traces}
              layout={{
                ...(panel.layout as Record<string, unknown>),
                height: halfHeight - 40,
                margin: { l: 40, r: 20, t: 10, b: 30 },
              }}
              height={halfHeight - 40}
              ariaLabel={panel.title}
            />
          </Card>
        ))}
      </div>
      {diagnostics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
          <StatBox label="Ljung-Box p" value={fmt(diagnostics.ljung_box_pvalue, 4)} warn={diagnostics.ljung_box_pvalue < 0.05} />
          <StatBox label="ARCH-LM p" value={fmt(diagnostics.arch_lm_pvalue, 4)} warn={diagnostics.has_arch_effects} />
          <StatBox label="Jarque-Bera p" value={fmt(diagnostics.jarque_bera_pvalue, 4)} warn={!diagnostics.is_normal} />
          <StatBox label="观测数" value={String(diagnostics.n_obs)} warn={false} />
        </div>
      )}
    </div>
  );
}

function StatBox({ label, value, warn }: { label: string; value: string; warn: boolean }) {
  return (
    <Card className={cn("p-3", warn && "border-chart-3/50")}>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={cn("text-lg font-mono font-bold", warn ? "text-chart-3" : "text-chart-2")}>{value}</div>
    </Card>
  );
}
