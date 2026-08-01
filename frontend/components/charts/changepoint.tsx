"use client";

import { PlotlyChart, chartColors } from "./plotly-chart";
import type { ChangepointSegment, TimePoint } from "@/lib/types";
import type { Data, Shape, Annotations } from "plotly.js";

interface ChangepointChartProps {
  spread: TimePoint[];
  breakpointDates: string[];
  segments: ChangepointSegment[];
  events?: { date: string; label: string }[];
  height?: number;
}

/**
 * Change point detection chart.
 * Displays raw spread with vertical dashed lines at breakpoints,
 * horizontal mean-level lines within each segment, and shaded
 * background areas highlighting each segment.
 */
export function ChangepointChart({
  spread,
  breakpointDates,
  segments,
  events = [],
  height = 400,
}: ChangepointChartProps) {
  const traces: Data[] = [];

  // Raw spread with hover template
  traces.push({
    x: spread.map((d) => d.date),
    y: spread.map((d) => d.value),
    type: "scatter" as const,
    mode: "lines" as const,
    name: "原始利差",
    line: { color: chartColors[0], width: 1 },
    opacity: 0.6,
    hovertemplate: "%{y:.2f} bps<extra>利差</extra>",
  });

  // Segment mean levels with text annotations and shaded background
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    const segColor = chartColors[(i % 4) + 1];

    // Shaded background area for the segment
    traces.push({
      x: [seg.start_date, seg.end_date, seg.end_date, seg.start_date],
      y: [
        seg.mean - (seg.std ?? 0),
        seg.mean - (seg.std ?? 0),
        seg.mean + (seg.std ?? 0),
        seg.mean + (seg.std ?? 0),
      ],
      type: "scatter" as const,
      mode: "lines" as const,
      fill: "toself" as const,
      fillcolor: segColor.replace(")", " / 0.10)"),
      line: { color: "transparent", width: 0 },
      showlegend: false,
      hoverinfo: "skip" as const,
    });

    // Mean level line (thicker)
    traces.push({
      x: [seg.start_date, seg.end_date],
      y: [seg.mean, seg.mean],
      type: "scatter" as const,
      mode: "lines" as const,
      name: `段 ${i + 1} (μ=${seg.mean.toFixed(1)})`,
      line: {
        color: segColor,
        width: 3,
      },
      showlegend: false,
      hovertemplate: `μ=${seg.mean.toFixed(2)} bps<extra>段 ${i + 1}</extra>`,
    });

    // Text annotation showing segment mean at midpoint
    const startDate = new Date(seg.start_date).getTime();
    const endDate = new Date(seg.end_date).getTime();
    const midDate = new Date((startDate + endDate) / 2).toISOString().split("T")[0];

    traces.push({
      x: [midDate],
      y: [seg.mean],
      type: "scatter" as const,
      mode: "text" as const,
      text: [`${seg.mean.toFixed(1)}`],
      textposition: "top center" as const,
      textfont: {
        size: 11,
        color: segColor,
        family: "monospace",
      },
      showlegend: false,
      hoverinfo: "skip" as const,
    });
  }

  // Shapes: vertical dashed lines at breakpoints (thicker, more visible)
  const shapes: Partial<Shape>[] = breakpointDates.map((d) => ({
    type: "line",
    x0: d,
    x1: d,
    yref: "paper" as const,
    y0: 0,
    y1: 1,
    line: {
      color: "oklch(0.55 0.15 25)",
      width: 2,
      dash: "dash" as const,
    },
  }));

  // Event annotations
  const annotations: Partial<Annotations>[] = events.map((e) => ({
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
