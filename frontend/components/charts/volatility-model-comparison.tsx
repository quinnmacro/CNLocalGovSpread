"use client";
import { PlotlyChart, chartColors } from "./plotly-chart";
import type { TimePoint } from "@/lib/types";
import type { Data } from "plotly.js";

interface ModelVol {
  name: string;
  volatility: TimePoint[];
}

interface VolatilityModelComparisonProps {
  models: ModelVol[];
  height?: number;
}

export function VolatilityModelComparison({
  models,
  height = 500,
}: VolatilityModelComparisonProps) {
  const traces: Data[] = models.map((m, i) => ({
    x: m.volatility.map((p) => p.date),
    y: m.volatility.map((p) => p.value),
    type: "scatter" as const,
    mode: "lines" as const,
    name: m.name,
    line: { color: chartColors[i % chartColors.length], width: 1.5 },
  }));

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: { title: { text: "日期" } },
        yaxis: { title: { text: "条件波动率 σ (bps)" } },
        legend: { orientation: "h" as const, y: -0.2 },
        hovermode: "x unified" as const,
      }}
      height={height}
      ariaLabel="多模型条件波动率对比图"
    />
  );
}
