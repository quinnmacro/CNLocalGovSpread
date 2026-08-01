"use client";
import { PlotlyChart, chartColors } from "./plotly-chart";
import type { TimePoint } from "@/lib/types";
import type { Data } from "plotly.js";

interface TimeSeriesChartProps {
  series: { name: string; data: TimePoint[]; color?: string }[];
  events?: { date: string; label: string }[];
  height?: number;
}

export function TimeSeriesChart({ series, events = [], height = 450 }: TimeSeriesChartProps) {
  const traces: Data[] = series.map((s, i) => ({
    x: s.data.map(d => d.date),
    y: s.data.map(d => d.value),
    type: "scatter" as const,
    mode: "lines" as const,
    name: s.name,
    line: { color: s.color ?? chartColors[i % chartColors.length], width: 1.5 },
    opacity: 0.9,
  }));

  // Add event annotations
  const shapes = events.map(e => ({
    type: "line" as const,
    x0: e.date, x1: e.date,
    y0: 0, y1: 1,
    yref: "paper" as const,
    line: { color: "oklch(0.5 0.15 25)", dash: "dash" as const, width: 1 },
  }));

  const annotations = events.map(e => ({
    x: e.date,
    y: 1,
    yref: "paper" as const,
    text: e.label,
    showarrow: false,
    font: { size: 10, color: "oklch(0.7 0.1 25)" },
    yanchor: "bottom" as const,
  }));

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: { title: { text: "日期" } },
        yaxis: { title: { text: "利差 (bps)" } },
        shapes,
        annotations,
        legend: { orientation: "h", y: -0.15 },
        hovermode: "x unified",
      }}
      height={height}
      ariaLabel="利差时序图"
    />
  );
}
