"use client";

import { PlotlyChart, chartColors } from "./plotly-chart";
import type { TimePoint } from "@/lib/types";
import type { Data, Shape } from "plotly.js";

interface KalmanSignalChartProps {
  spread: TimePoint[];
  signal: TimePoint[];
  deviationZscore: TimePoint[];
  events?: { date: string; label: string }[];
  height?: number;
}

/**
 * Dual-panel Kalman filter signal extraction chart.
 * Top: raw spread + Kalman smoothed trend.
 * Bottom: deviation z-score with ±1.5 threshold bands.
 */
export function KalmanSignalChart({
  spread,
  signal,
  deviationZscore,
  events = [],
  height = 520,
}: KalmanSignalChartProps) {
  // ── Top panel traces ──────────────────────────────────────────────
  const topTraces: Data[] = [
    {
      x: spread.map((d) => d.date),
      y: spread.map((d) => d.value),
      type: "scatter" as const,
      mode: "lines" as const,
      name: "原始利差",
      line: { color: chartColors[0], width: 1 },
      opacity: 0.6,
      xaxis: "x",
      yaxis: "y",
    },
    {
      x: signal.map((d) => d.date),
      y: signal.map((d) => d.value),
      type: "scatter" as const,
      mode: "lines" as const,
      name: "Kalman 信号（趋势）",
      line: { color: chartColors[2], width: 2 },
      xaxis: "x",
      yaxis: "y",
    },
  ];

  // ── Bottom panel traces ───────────────────────────────────────────
  const zDates = deviationZscore.map((d) => d.date);
  const zValues = deviationZscore.map((d) => d.value);

  const bottomTraces: Data[] = [
    {
      x: zDates,
      y: zValues,
      type: "scatter" as const,
      mode: "lines" as const,
      name: "偏离度 z-score",
      line: { color: chartColors[5], width: 1.5 },
      fill: "tozeroy" as const,
      fillcolor: "rgba(100,180,255,0.08)",
      xaxis: "x2",
      yaxis: "y2",
    },
    // +1.5 threshold
    {
      x: zDates,
      y: zDates.map(() => 1.5),
      type: "scatter" as const,
      mode: "lines" as const,
      name: "+1.5 阈值（高估）",
      line: { color: "oklch(0.65 0.15 25)", width: 1, dash: "dash" as const },
      xaxis: "x2",
      yaxis: "y2",
    },
    // -1.5 threshold
    {
      x: zDates,
      y: zDates.map(() => -1.5),
      type: "scatter" as const,
      mode: "lines" as const,
      name: "−1.5 阈值（低估）",
      line: { color: "oklch(0.65 0.15 140)", width: 1, dash: "dash" as const },
      xaxis: "x2",
      yaxis: "y2",
    },
  ];

  const traces: Data[] = [...topTraces, ...bottomTraces];

  // ── Background shading for z-score extremes ──────────────────────
  const shapes: Partial<Shape>[] = [
    // Overvalued zone (z > 1.5)
    {
      type: "rect",
      xref: "paper",
      x0: 0,
      x1: 1,
      y0: 1.5,
      y1: Math.max(3, ...zValues),
      yref: "y2",
      fillcolor: "oklch(0.45 0.15 25)",
      opacity: 0.08,
      line: { width: 0 },
    },
    // Undervalued zone (z < -1.5)
    {
      type: "rect",
      xref: "paper",
      x0: 0,
      x1: 1,
      y0: Math.min(-3, ...zValues),
      y1: -1.5,
      yref: "y2",
      fillcolor: "oklch(0.45 0.15 140)",
      opacity: 0.08,
      line: { width: 0 },
    },
  ];

  // ── Event annotations (on top panel) ─────────────────────────────
  const annotations = events.map((e) => ({
    x: e.date,
    y: 1.02,
    yref: "paper" as const,
    text: e.label,
    showarrow: true,
    arrowhead: 2,
    arrowcolor: "oklch(0.5 0.15 25)",
    ax: 0,
    ay: -25,
    font: { size: 9, color: "oklch(0.7 0.1 25)" },
  }));

  return (
    <PlotlyChart
      data={traces}
      layout={{
        shapes,
        annotations,
        grid: { rows: 2, columns: 1, pattern: "independent", subplots: [["xy"], ["x2y2"]] },
        xaxis: {
          title: { text: "" },
          rangeslider: { visible: false },
        },
        yaxis: {
          title: { text: "利差 (bps)" },
          domain: [0.52, 1],
        },
        xaxis2: {
          title: { text: "日期" },
          matches: "x",
        },
        yaxis2: {
          title: { text: "z-score" },
          domain: [0, 0.45],
        },
        legend: { orientation: "h", y: -0.12 },
        hovermode: "x unified",
      }}
      height={height}
      ariaLabel="Kalman 滤波信号提取图"
    />
  );
}
