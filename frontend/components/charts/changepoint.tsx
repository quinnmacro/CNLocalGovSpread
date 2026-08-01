"use client";

import { PlotlyChart, chartColors } from "./plotly-chart";
import type { ChangepointSegment, TimePoint } from "@/lib/types";
import type { Data, Shape } from "plotly.js";

interface ChangepointChartProps {
  spread: TimePoint[];
  breakpointDates: string[];
  segments: ChangepointSegment[];
  events?: { date: string; label: string }[];
  height?: number;
}

/**
 * Change point detection chart.
 * Displays raw spread with vertical dashed lines at breakpoints
 * and horizontal mean-level lines within each segment.
 */
export function ChangepointChart({
  spread,
  breakpointDates,
  segments,
  events = [],
  height = 400,
}: ChangepointChartProps) {
  const traces: Data[] = [];

  // Raw spread
  traces.push({
    x: spread.map((d) => d.date),
    y: spread.map((d) => d.value),
    type: "scatter" as const,
    mode: "lines" as const,
    name: "原始利差",
    line: { color: chartColors[0], width: 1 },
    opacity: 0.6,
  });

  // Segment mean levels
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    traces.push({
      x: [seg.start_date, seg.end_date],
      y: [seg.mean, seg.mean],
      type: "scatter" as const,
      mode: "lines" as const,
      name: `段 ${i + 1} (μ=${seg.mean.toFixed(1)})`,
      line: {
        color: chartColors[(i % 4) + 1],
        width: 2.5,
      },
      showlegend: false,
    });
  }

  // Shapes: vertical dashed lines at breakpoints
  const shapes: Partial<Shape>[] = breakpointDates.map((d) => ({
    type: "line",
    x0: d,
    x1: d,
    yref: "paper" as const,
    y0: 0,
    y1: 1,
    line: {
      color: "oklch(0.60 0.12 25)",
      width: 1.5,
      dash: "dash" as const,
    },
  }));

  // Event annotations
  const annotations = events.map((e) => ({
    x: e.date,
    y: 1.03,
    yref: "paper" as const,
    text: e.label,
    showarrow: true,
    arrowhead: 2,
    arrowcolor: "oklch(0.5 0.15 25)",
    ax: 0,
    ay: -30,
    font: { size: 9, color: "oklch(0.7 0.1 25)" },
  }));

  return (
    <PlotlyChart
      data={traces}
      layout={{
        shapes,
        annotations,
        xaxis: { title: { text: "日期" } },
        yaxis: { title: { text: "利差 (bps)" } },
        hovermode: "x unified",
        legend: { orientation: "h", y: -0.2 },
      }}
      height={height}
      ariaLabel="结构性变化点检测图"
    />
  );
}
