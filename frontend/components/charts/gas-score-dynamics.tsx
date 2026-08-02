"use client";
import { PlotlyChart, chartColors } from "./plotly-chart";
import type { TimePoint } from "@/lib/types";
import type { Data } from "plotly.js";

interface GASScoreDynamicsProps {
  condVol: TimePoint[];
  scoreSeries: TimePoint[];
  height?: number;
}

export function GASScoreDynamics({
  condVol,
  scoreSeries,
  height = 480,
}: GASScoreDynamicsProps) {
  const traces: Data[] = [
    {
      x: condVol.map((p) => p.date),
      y: condVol.map((p) => p.value),
      type: "scatter" as const,
      mode: "lines" as const,
      name: "GAS 条件波动率",
      line: { color: chartColors[0], width: 2 },
      yaxis: "y",
    },
    {
      x: scoreSeries.map((p) => p.date),
      y: scoreSeries.map((p) => p.value),
      type: "bar" as const,
      name: "Score 更新量",
      marker: {
        color: scoreSeries.map((p) =>
          p.value >= 0 ? "rgba(34,197,94,0.5)" : "rgba(239,68,68,0.5)"
        ),
      },
      yaxis: "y2",
    },
  ];

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: { title: { text: "日期" } },
        yaxis: {
          title: { text: "条件波动率" },
          domain: [0.35, 1],
        },
        yaxis2: {
          title: { text: "Score" },
          overlaying: "y" as const,
          side: "right" as const,
          domain: [0, 0.25],
          showgrid: false,
        },
        legend: { orientation: "h" as const, y: -0.15 },
        hovermode: "x unified" as const,
      }}
      height={height}
      ariaLabel="GAS 波动率与 Score 动态图"
    />
  );
}
