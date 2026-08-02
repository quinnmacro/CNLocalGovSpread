"use client";
import { PlotlyChart, chartColors } from "./plotly-chart";
import type { TimePoint } from "@/lib/types";
import type { Data } from "plotly.js";

interface StochasticVolBandProps {
  condVol: TimePoint[];
  volLower: TimePoint[];
  volUpper: TimePoint[];
  logVol?: TimePoint[];
  height?: number;
}

export function StochasticVolBand({
  condVol,
  volLower,
  volUpper,
  logVol,
  height = 420,
}: StochasticVolBandProps) {
  const dates = condVol.map((p) => p.date);

  const traces: Data[] = [
    // Upper band (invisible line for fill)
    {
      x: dates,
      y: volUpper.map((p) => p.value),
      type: "scatter" as const,
      mode: "lines" as const,
      line: { color: "transparent" },
      name: "80% CI 上界",
      showlegend: false,
      hoverinfo: "skip" as const,
    },
    // Lower band + fill to upper
    {
      x: dates,
      y: volLower.map((p) => p.value),
      type: "scatter" as const,
      mode: "lines" as const,
      fill: "tonexty" as const,
      fillcolor: "rgba(99, 179, 237, 0.15)",
      line: { color: "transparent" },
      name: "80% CI 下界",
      showlegend: true,
    },
    // Posterior mean
    {
      x: dates,
      y: condVol.map((p) => p.value),
      type: "scatter" as const,
      mode: "lines" as const,
      name: "后验均值 σ",
      line: { color: chartColors[1], width: 2 },
    },
  ];

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: { title: { text: "日期" } },
        yaxis: { title: { text: "条件波动率" } },
        legend: { orientation: "h" as const, y: -0.2 },
        hovermode: "x unified" as const,
      }}
      height={height}
      ariaLabel="随机波动率后验区间图"
    />
  );
}
