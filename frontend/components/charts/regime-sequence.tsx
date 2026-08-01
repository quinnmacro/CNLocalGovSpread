"use client";
import { PlotlyChart, chartColors } from "./plotly-chart";
import type { RegimeLabel, TimePoint } from "@/lib/types";
import type { Data, Shape } from "plotly.js";

interface RegimeSequenceChartProps {
  labels: RegimeLabel[];
  volatility?: TimePoint[];
  spread?: TimePoint[];
  nRegimes: number;
  events?: { date: string; label: string }[];
  height?: number;
}

const regimeColors = [
  "oklch(0.55 0.12 170)", // low vol - teal
  "oklch(0.65 0.15 250)", // mid vol - blue
  "oklch(0.55 0.2 25)",   // high vol - red
];

export function RegimeSequenceChart({
  labels,
  volatility,
  spread,
  nRegimes,
  events = [],
  height = 400,
}: RegimeSequenceChartProps) {
  const traces: Data[] = [];

  // Background colored bands for regimes
  const shapes: Partial<Shape>[] = [];
  if (labels.length > 1) {
    for (let i = 0; i < labels.length - 1; i++) {
      shapes.push({
        type: "rect",
        x0: labels[i].date,
        x1: labels[i + 1].date,
        y0: 0,
        y1: 1,
        yref: "paper",
        fillcolor: regimeColors[labels[i].regime % regimeColors.length],
        opacity: 0.15,
        line: { width: 0 },
      });
    }
  }

  // Volatility or spread line
  const lineData = volatility ?? spread;
  if (lineData && lineData.length > 0) {
    traces.push({
      x: lineData.map(d => d.date),
      y: lineData.map(d => d.value),
      type: "scatter" as const,
      mode: "lines" as const,
      name: volatility ? "波动率" : "利差",
      line: { color: chartColors[0], width: 1.5 },
    });
  }

  // Event annotations
  const annotations = events.map(e => ({
    x: e.date,
    y: 1.02,
    yref: "paper" as const,
    text: e.label,
    showarrow: true,
    arrowhead: 2,
    arrowcolor: "oklch(0.5 0.15 25)",
    ax: 0,
    ay: -30,
    font: { size: 10, color: "oklch(0.7 0.1 25)" },
  }));

  // Legend for regimes
  for (let r = 0; r < nRegimes; r++) {
    const regimeNames = ["低波动", "中波动", "高波动"];
    traces.push({
      x: [null],
      y: [null],
      type: "scatter" as const,
      mode: "markers" as const,
      name: `状态 ${r} (${regimeNames[r] ?? r})`,
      marker: { color: regimeColors[r % regimeColors.length], size: 10 },
    });
  }

  return (
    <PlotlyChart
      data={traces}
      layout={{
        shapes,
        xaxis: { title: { text: "日期" } },
        yaxis: { title: { text: volatility ? "波动率" : "利差 (bps)" } },
        annotations,
        legend: { orientation: "h", y: -0.2 },
        hovermode: "x unified",
      }}
      height={height}
      ariaLabel="HMM 状态时序图"
    />
  );
}
