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
  chartColors[1], // low vol - teal
  chartColors[0], // mid vol - blue
  chartColors[6], // high vol - red
];

const regimeNames = ["低波动 (Calm)", "中波动 (Caution)", "高波动 (Stress)"];

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
      const color = regimeColors[labels[i].regime % regimeColors.length];
      shapes.push({
        type: "rect",
        x0: labels[i].date,
        x1: labels[i + 1].date,
        y0: 0,
        y1: 1,
        yref: "paper",
        fillcolor: color.replace(")", " / 0.12)"),
        opacity: 0.9,
        line: { width: 0 },
      });
    }
  }

  // Volatility or spread line
  const lineData = volatility ?? spread;
  const isVol = !!volatility;
  if (lineData && lineData.length > 0) {
    traces.push({
      x: lineData.map((d) => d.date),
      y: lineData.map((d) => d.value),
      type: "scatter" as const,
      mode: "lines" as const,
      name: isVol ? "波动率 (σ)" : "原始利差",
      line: { color: chartColors[0], width: 1.5 },
      hovertemplate: isVol
        ? "%{y:.4f}<extra>波动率</extra>"
        : "%{y:.2f} bps<extra>利差</extra>",
    });

    // Annotation for the last data point
    const lastPt = lineData[lineData.length - 1];
    traces.push({
      x: [lastPt.date],
      y: [lastPt.value],
      type: "scatter" as const,
      mode: "text" as const,
      text: [isVol ? `σ=${lastPt.value.toFixed(4)}` : `${lastPt.value.toFixed(1)}`],
      textposition: "top right" as const,
      textfont: { size: 10, color: chartColors[5], family: "monospace" },
      showlegend: false,
      hoverinfo: "skip" as const,
    });
  }

  // Event annotations
  const annotations = events.map((e) => ({
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

  // Legend entries for regimes (invisible traces for legend display)
  for (let r = 0; r < nRegimes; r++) {
    traces.push({
      x: [null],
      y: [null],
      type: "scatter" as const,
      mode: "markers" as const,
      name: `状态 ${r}: ${regimeNames[r % regimeNames.length]}`,
      marker: {
        color: regimeColors[r % regimeColors.length],
        size: 10,
        symbol: "square",
      },
      hoverinfo: "skip" as const,
    });
  }

  return (
    <PlotlyChart
      data={traces}
      layout={{
        shapes,
        xaxis: { title: { text: "日期" } },
        yaxis: { title: { text: isVol ? "波动率 (σ)" : "利差 (bps)" } },
        annotations,
        legend: {
          orientation: "h",
          y: -0.18,
          x: 0,
          xanchor: "left",
          bgcolor: "rgba(0,0,0,0)",
          font: { size: 11 },
        },
        hovermode: "x unified",
        margin: { l: 55, r: 30, t: 30, b: 55 },
      }}
      height={height}
      ariaLabel="HMM 状态时序图"
    />
  );
}
