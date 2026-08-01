"use client";
import { PlotlyChart } from "./plotly-chart";
import type { Data } from "plotly.js";
import { fmt } from "@/lib/utils";

interface TransitionHeatmapProps {
  matrix: number[][];
  regimeNames?: string[];
  height?: number;
}

export function TransitionHeatmap({
  matrix,
  regimeNames,
  height = 350,
}: TransitionHeatmapProps) {
  const n = matrix.length;
  const labels =
    regimeNames ?? Array.from({ length: n }, (_, i) => `状态 ${i}`);

  // Format values as text annotations
  const text = matrix.map((row) => row.map((v) => fmt(v, 3)));

  const traces: Data[] = [
    {
      z: matrix,
      x: labels,
      y: labels,
      type: "heatmap" as const,
      colorscale: [
        [0, "oklch(0.15 0.02 260)"],
        [0.3, "oklch(0.3 0.08 250)"],
        [0.6, "oklch(0.5 0.12 250)"],
        [1, "oklch(0.7 0.15 250)"],
      ],
      text,
      texttemplate: "%{text}",
      textfont: { size: 15, color: "white", family: "var(--font-mono, monospace)" },
      hovertemplate: "%{y} → %{x}<br>概率: %{z:.3f}<extra></extra>",
      showscale: true,
      xgap: 3,
      ygap: 3,
      colorbar: {
        title: { text: "转移概率", font: { size: 11 } },
        tickvals: [0, 0.25, 0.5, 0.75, 1],
        tickfont: { size: 10 },
        len: 0.8,
        thickness: 12,
        outlinewidth: 0,
        borderwidth: 0,
      },
    } as unknown as Data,
  ];

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: {
          title: { text: "转移到", font: { size: 12 } },
          side: "bottom" as const,
          tickfont: { size: 11 },
        },
        yaxis: {
          title: { text: "当前状态", font: { size: 12 } },
          autorange: "reversed" as const,
          tickfont: { size: 11 },
        },
        margin: { l: 80, r: 50, t: 30, b: 60 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
      }}
      height={height}
      ariaLabel="转移概率矩阵热力图"
    />
  );
}
