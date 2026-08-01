"use client";
import { PlotlyChart, chartColors } from "./plotly-chart";
import type { Data } from "plotly.js";

interface FanChartProps {
  currentSpread: number;
  horizon: number;
  median: number;
  p5: number;
  p95: number;
  height?: number;
}

export function FanChart({ currentSpread, horizon, median, p5, p95, height = 450 }: FanChartProps) {
  // Generate fan chart with approximate quantile bands
  // Since the API returns final quantiles, we interpolate linearly
  const steps = 50;
  const x = Array.from({ length: steps + 1 }, (_, i) => i);
  const t = x.map(i => i / steps);

  const medianLine = t.map(ti => currentSpread + (median - currentSpread) * ti);
  const upperLine = t.map(ti => currentSpread + (p95 - currentSpread) * Math.sqrt(ti));
  const lowerLine = t.map(ti => currentSpread + (p5 - currentSpread) * Math.sqrt(ti));

  // Intermediate bands (approximate p25/p75)
  const p25Line = t.map((ti, i) => (medianLine[i] + lowerLine[i]) / 2);
  const p75Line = t.map((ti, i) => (medianLine[i] + upperLine[i]) / 2);

  const traces: Data[] = [
    // 90% band
    {
      x: [...x, ...x.slice().reverse()],
      y: [...upperLine, ...lowerLine.slice().reverse()],
      type: "scatter" as const,
      fill: "toself" as const,
      fillcolor: `${chartColors[0]}20`,
      line: { color: "transparent" },
      name: "90% 区间",
      showlegend: true,
    },
    // 50% band
    {
      x: [...x, ...x.slice().reverse()],
      y: [...p75Line, ...p25Line.slice().reverse()],
      type: "scatter" as const,
      fill: "toself" as const,
      fillcolor: `${chartColors[0]}40`,
      line: { color: "transparent" },
      name: "50% 区间",
      showlegend: true,
    },
    // Median
    {
      x,
      y: medianLine,
      type: "scatter" as const,
      mode: "lines" as const,
      name: "中位数",
      line: { color: chartColors[0], width: 2 },
    },
  ];

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: { title: { text: `交易日 (horizon = ${horizon})` } },
        yaxis: { title: { text: "利差 (bps)" } },
        legend: { orientation: "h", y: -0.15 },
        hovermode: "x unified",
      }}
      height={height}
      ariaLabel="蒙特卡洛扇形图"
    />
  );
}
