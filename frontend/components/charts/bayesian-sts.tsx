"use client";
import { PlotlyChart, chartColors } from "./plotly-chart";
import type { TimePoint } from "@/lib/types";
import type { Data } from "plotly.js";

interface BayesianSTSChartProps {
  spread: TimePoint[];
  signal: TimePoint[];
  signalLower: TimePoint[];
  signalUpper: TimePoint[];
  deviationZscore: TimePoint[];
  events?: { date: string; label: string }[];
  height?: number;
}

const PURPLE = chartColors[4]; // Use chart palette for consistency

export function BayesianSTSChart({
  spread,
  signal,
  signalLower,
  signalUpper,
  deviationZscore,
  events = [],
  height = 680,
}: BayesianSTSChartProps) {
  const dates = spread.map((p) => p.date);
  const lastZscore = deviationZscore[deviationZscore.length - 1];
  const lastDate = dates[dates.length - 1];

  const upperFill: Data = {
    x: dates,
    y: signalUpper.map((p) => p.value),
    type: "scatter",
    mode: "lines",
    line: { color: "transparent", width: 0 },
    showlegend: false,
    hoverinfo: "skip",
  };

  const lowerFill: Data = {
    x: dates,
    y: signalLower.map((p) => p.value),
    type: "scatter",
    mode: "lines",
    fill: "tonexty",
    fillcolor: "rgba(120,200,160,0.15)",
    line: { color: "transparent", width: 0 },
    name: "80% 置信区间",
    hovertemplate: "CI 下界: %{y:.2f} bps<extra>80% CI</extra>",
  };

  const spreadTrace: Data = {
    x: dates,
    y: spread.map((p) => p.value),
    type: "scatter",
    mode: "lines",
    name: "原始利差",
    line: { color: chartColors[0], width: 1 },
    opacity: 0.5,
    hovertemplate: "%{y:.2f} bps<extra>%{fullData.name}</extra>",
  };

  const signalTrace: Data = {
    x: dates,
    y: signal.map((p) => p.value),
    type: "scatter",
    mode: "lines",
    name: "Bayesian 趋势",
    line: { color: chartColors[1], width: 2.5 },
    hovertemplate: "%{y:.2f} bps<extra>%{fullData.name}</extra>",
  };

  const zscoreTrace: Data = {
    x: deviationZscore.map((p) => p.date),
    y: deviationZscore.map((p) => p.value),
    type: "scatter",
    mode: "lines",
    name: "偏离度 z-score",
    line: { color: PURPLE, width: 1.5 },
    xaxis: "x2",
    yaxis: "y2",
    hovertemplate: "z = %{y:+.3f} σ<extra>%{fullData.name}</extra>",
  };

  const zscoreZero: Data = {
    x: [deviationZscore[0]?.date, lastDate],
    y: [0, 0],
    type: "scatter",
    mode: "lines",
    line: { color: "#888", width: 1, dash: "dash" },
    showlegend: false,
    xaxis: "x2",
    yaxis: "y2",
    hoverinfo: "skip",
  };

  const traces: Data[] = [
    upperFill,
    lowerFill,
    spreadTrace,
    signalTrace,
    zscoreTrace,
    zscoreZero,
  ];

  const shapes = events.map((ev) => ({
    type: "line" as const,
    x0: ev.date,
    x1: ev.date,
    y0: 0,
    y1: 1,
    yref: "paper" as const,
    line: { color: "rgba(200,200,200,0.3)", width: 1, dash: "dot" as const },
  }));

  const lastZVal = lastZscore?.value ?? 0;
  const isExtreme = Math.abs(lastZVal) >= 1.5;
  const zColor = isExtreme ? chartColors[6] : PURPLE;

  const annotations = lastZscore
    ? [
        {
          x: lastDate,
          y: lastZVal,
          xanchor: "left" as const,
          yanchor: "bottom" as const,
          text: `z = ${lastZVal >= 0 ? "+" : ""}${lastZVal.toFixed(3)}`,
          showarrow: false,
          font: { size: 12, color: zColor, weight: 600 as const },
          xref: "x2" as const,
          yref: "y2" as const,
          xshift: 8,
          bgcolor: "rgba(0,0,0,0.6)",
          bordercolor: zColor,
          borderwidth: 1,
          borderpad: 4,
        },
      ]
    : [];

  return (
    <PlotlyChart
      data={traces}
      layout={{
        height,
        grid: {
          rows: 2,
          columns: 1,
          pattern: "independent",
          roworder: "top to bottom",
        },
        xaxis: { type: "date", title: { text: "" } },
        yaxis: { title: { text: "利差 (bps)" }, domain: [0.55, 1] },
        xaxis2: { type: "date", title: { text: "日期" }, matches: "x" },
        yaxis2: {
          title: { text: "z-score" },
          zeroline: false,
          domain: [0, 0.4],
        },
        legend: {
          orientation: "h",
          x: 0,
          y: 1.08,
          xanchor: "left",
          yanchor: "bottom",
          bgcolor: "rgba(0,0,0,0)",
          font: { size: 11 },
        },
        margin: { l: 55, r: 70, t: 40, b: 40 },
        hovermode: "x unified",
        shapes,
        annotations,
      }}
      ariaLabel="Bayesian STS 信号与偏差图"
    />
  );
}
