"use client";

import { PlotlyChart, chartColors } from "./plotly-chart";
import type { TimePoint } from "@/lib/types";
import type { Data, Shape } from "plotly.js";

interface STSSignalChartProps {
  spread: TimePoint[];
  signal: TimePoint[];
  slope: TimePoint[];
  deviationZscore: TimePoint[];
  events?: { date: string; label: string }[];
  height?: number;
}

/**
 * Structural Time Series signal chart.
 * Top: raw spread + STS smoothed level (signal).
 * Middle: slope (drift/speed of trend change).
 * Bottom: deviation z-score with ±1.5 threshold bands.
 */
export function STSSignalChart({
  spread,
  signal,
  slope,
  deviationZscore,
  events = [],
  height = 620,
}: STSSignalChartProps) {
  const topTraces: Data[] = [
    {
      x: spread.map((d) => d.date),
      y: spread.map((d) => d.value),
      type: "scatter" as const,
      mode: "lines" as const,
      name: "原始利差",
      line: { color: chartColors[0], width: 1 },
      opacity: 0.5,
      xaxis: "x",
      yaxis: "y",
      hovertemplate: "%{y:.2f} bps<extra>%{fullData.name}</extra>",
    },
    {
      x: signal.map((d) => d.date),
      y: signal.map((d) => d.value),
      type: "scatter" as const,
      mode: "lines" as const,
      name: "STS 趋势 (level)",
      line: { color: chartColors[2], width: 2.5 },
      xaxis: "x",
      yaxis: "y",
      hovertemplate: "%{y:.2f} bps<extra>%{fullData.name}</extra>",
    },
  ];

  const slopeTraces: Data[] = [
    {
      x: slope.map((d) => d.date),
      y: slope.map((d) => d.value),
      type: "scatter" as const,
      mode: "lines" as const,
      name: "漂移率 (slope)",
      line: { color: chartColors[4], width: 1.5 },
      fill: "tozeroy" as const,
      fillcolor: "rgba(180,200,100,0.12)",
      xaxis: "x2",
      yaxis: "y2",
      hovertemplate: "%{y:+.4f}<extra>%{fullData.name}</extra>",
    },
  ];

  const zDates = deviationZscore.map((d) => d.date);
  const zValues = deviationZscore.map((d) => d.value);
  const lastZ = zValues.length > 0 ? zValues[zValues.length - 1] : 0;
  const lastZDate = zDates.length > 0 ? zDates[zDates.length - 1] : "";

  const bottomTraces: Data[] = [
    {
      x: zDates,
      y: zValues,
      type: "scatter" as const,
      mode: "lines" as const,
      name: "偏离度 z-score",
      line: { color: chartColors[5], width: 1.5 },
      fill: "tozeroy" as const,
      fillcolor: "rgba(100,180,220,0.10)",
      xaxis: "x3",
      yaxis: "y3",
      hovertemplate: "%{y:+.3f} σ<extra>%{fullData.name}</extra>",
    },
    {
      x: zDates,
      y: zDates.map(() => 1.5),
      type: "scatter" as const,
      mode: "lines" as const,
      name: "+1.5 阈值（高估）",
      line: { color: chartColors[6], width: 1, dash: "dash" as const },
      xaxis: "x3",
      yaxis: "y3",
      hovertemplate: "%{y:+.1f} σ<extra>%{fullData.name}</extra>",
    },
    {
      x: zDates,
      y: zDates.map(() => -1.5),
      type: "scatter" as const,
      mode: "lines" as const,
      name: "−1.5 阈值（低估）",
      line: { color: chartColors[1], width: 1, dash: "dash" as const },
      xaxis: "x3",
      yaxis: "y3",
      hovertemplate: "%{y:+.1f} σ<extra>%{fullData.name}</extra>",
    },
  ];

  const traces: Data[] = [...topTraces, ...slopeTraces, ...bottomTraces];

  const shapes: Partial<Shape>[] = [
    {
      type: "rect",
      xref: "paper",
      x0: 0,
      x1: 1,
      y0: 1.5,
      y1: Math.max(3, ...zValues),
      yref: "y3",
      fillcolor: "rgba(255,100,80,0.06)",
      opacity: 0.8,
      line: { width: 0 },
    },
    {
      type: "rect",
      xref: "paper",
      x0: 0,
      x1: 1,
      y0: Math.min(-3, ...zValues),
      y1: -1.5,
      yref: "y3",
      fillcolor: "rgba(80,200,120,0.06)",
      opacity: 0.8,
      line: { width: 0 },
    },
  ];

  const annotations = [
    // Event annotations
    ...events.map((e) => ({
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
    })),
    // Current z-score annotation on right side of bottom panel
    ...(lastZDate
      ? [
          {
            x: lastZDate,
            y: lastZ,
            xref: "x3" as const,
            yref: "y3" as const,
            text: `z = ${lastZ >= 0 ? "+" : ""}${lastZ.toFixed(3)}`,
            showarrow: true,
            arrowhead: 0,
            arrowcolor: Math.abs(lastZ) >= 1.5 ? chartColors[6] : chartColors[5],
            ax: 45,
            ay: 0,
            font: {
              size: 11,
              color: Math.abs(lastZ) >= 1.5 ? chartColors[6] : chartColors[5],
              weight: 600 as const,
            },
            bgcolor: "rgba(0,0,0,0.6)",
            bordercolor: Math.abs(lastZ) >= 1.5 ? chartColors[6] : chartColors[5],
            borderwidth: 1,
            borderpad: 4,
          },
        ]
      : []),
  ];

  return (
    <PlotlyChart
      data={traces}
      layout={{
        shapes,
        annotations,
        xaxis: { title: { text: "" }, domain: [0, 1], anchor: "y" },
        yaxis: { title: { text: "利差 (bps)" }, domain: [0.65, 1] },
        xaxis2: { title: { text: "" }, domain: [0, 1], anchor: "y2", matches: "x" },
        yaxis2: { title: { text: "slope" }, domain: [0.38, 0.58] },
        xaxis3: { title: { text: "日期" }, domain: [0, 1], anchor: "y3", matches: "x" },
        yaxis3: { title: { text: "z-score" }, domain: [0, 0.3] },
        legend: {
          orientation: "h",
          x: 0,
          xanchor: "left",
          y: -0.12,
          yanchor: "top",
          bgcolor: "rgba(0,0,0,0)",
          font: { size: 11 },
          traceorder: "normal",
        },
        hovermode: "x unified",
        margin: { l: 55, r: 60, t: 30, b: 55 },
      }}
      height={height}
      ariaLabel="Structural Time Series 信号提取图"
    />
  );
}
