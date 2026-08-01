"use client";
import { PlotlyChart, chartColors } from "./plotly-chart";
import type { RiskMetrics } from "@/lib/types";
import type { Data } from "plotly.js";
import { fmt } from "@/lib/utils";

interface VarComparisonChartProps {
  metrics: RiskMetrics;
  confidence: number;
  height?: number;
}

export function VarComparisonChart({ metrics, confidence, height = 350 }: VarComparisonChartProps) {
  const methods = ["历史模拟", "参数法", "EVT"];
  const varValues = [metrics.var_historical, metrics.var_parametric, metrics.var_evt];
  const esValue = metrics.es_evt;

  const traces: Data[] = [
    {
      x: methods,
      y: varValues,
      type: "bar" as const,
      name: `VaR (${fmt(confidence * 100, 1)}%)`,
      marker: { color: chartColors[0] },
      text: varValues.map(v => `${fmt(v, 2)} bps`),
      textposition: "outside" as const,
    },
    {
      x: ["EVT"],
      y: [esValue],
      type: "bar" as const,
      name: "ES (EVT)",
      marker: { color: chartColors[2] },
      text: [`${fmt(esValue, 2)} bps`],
      textposition: "outside" as const,
    },
  ];

  return (
    <PlotlyChart
      data={traces}
      layout={{
        yaxis: { title: { text: "风险值 (bps)" } },
        barmode: "group" as const,
        legend: { orientation: "h", y: -0.2 },
      }}
      height={height}
      ariaLabel="VaR 方法对比图"
    />
  );
}
