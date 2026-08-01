"use client";
import { PlotlyChart, chartColors } from "./plotly-chart";
import type { HillInfo } from "@/lib/types";
import type { Data } from "plotly.js";
import { fmt } from "@/lib/utils";

interface HillPlotProps {
  hill: HillInfo;
  height?: number;
}

export function HillPlot({ hill, height = 350 }: HillPlotProps) {
  const traces: Data[] = [
    {
      x: hill.estimates.map(e => e.k_percentile),
      y: hill.estimates.map(e => e.tail_index),
      type: "scatter" as const,
      mode: "lines+markers" as const,
      name: "Hill 估计",
      line: { color: chartColors[0], width: 1.5 },
      marker: { size: 3 },
    },
    // Reference line at selected tail index
    {
      x: [hill.estimates[0]?.k_percentile ?? 0, hill.estimates[hill.estimates.length - 1]?.k_percentile ?? 1],
      y: [hill.tail_index, hill.tail_index],
      type: "scatter" as const,
      mode: "lines" as const,
      name: `选定 ξ = ${fmt(hill.tail_index, 3)}`,
      line: { color: chartColors[2], dash: "dash" as const, width: 1.5 },
    },
  ];

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: { title: { text: "k 分位数 (%)", range: [0, 0.2] } },
        yaxis: { title: { text: "尾指数 (ξ)" } },
        legend: { orientation: "h", y: -0.2 },
        hovermode: "x unified",
        annotations: [
          {
            x: 0.98,
            y: 0.95,
            xref: "paper" as const,
            yref: "paper" as const,
            text: `threshold = ${fmt(hill.threshold, 3)}`,
            showarrow: false,
            font: { size: 11, color: "oklch(0.65 0.01 260)" },
            align: "right" as const,
          },
        ],
      }}
      height={height}
      ariaLabel="Hill 尾指数估计图"
    />
  );
}
