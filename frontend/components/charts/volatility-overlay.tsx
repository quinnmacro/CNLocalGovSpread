"use client";
import { PlotlyChart, chartColors } from "./plotly-chart";
import type { TimePoint } from "@/lib/types";
import type { Data } from "plotly.js";

interface VolatilityOverlayProps {
  models: { name: string; volatility: TimePoint[]; color?: string }[];
  returns?: TimePoint[];
  height?: number;
}

export function VolatilityOverlay({ models, returns, height = 450 }: VolatilityOverlayProps) {
  const traces: Data[] = [];

  // Returns as bar chart at bottom
  if (returns && returns.length > 0) {
    traces.push({
      x: returns.map(r => r.date),
      y: returns.map(r => r.value),
      type: "bar" as const,
      name: "日收益率",
      marker: { color: "oklch(0.5 0.01 260)", opacity: 0.3 },
      yaxis: "y2",
      showlegend: true,
    });
  }

  // Volatility lines
  models.forEach((m, i) => {
    traces.push({
      x: m.volatility.map(v => v.date),
      y: m.volatility.map(v => v.value),
      type: "scatter" as const,
      mode: "lines" as const,
      name: m.name,
      line: { color: m.color ?? chartColors[i % chartColors.length], width: 1.5 },
    });
  });

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: { title: { text: "日期" } },
        yaxis: { title: { text: "条件波动率 (bps)" }, domain: [0.3, 1] },
        yaxis2: {
          title: { text: "日收益率" },
          overlaying: "y",
          side: "right",
          domain: [0, 0.25],
          showgrid: false,
        },
        legend: { orientation: "h", y: -0.2 },
        hovermode: "x unified",
      }}
      height={height}
      ariaLabel="条件波动率对比图"
    />
  );
}
