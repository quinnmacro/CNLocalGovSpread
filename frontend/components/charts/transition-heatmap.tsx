"use client";
import { PlotlyChart } from "./plotly-chart";
import type { Data } from "plotly.js";
import { fmt } from "@/lib/utils";

interface TransitionHeatmapProps {
  matrix: number[][];
  regimeNames?: string[];
  height?: number;
}

export function TransitionHeatmap({ matrix, regimeNames, height = 350 }: TransitionHeatmapProps) {
  const n = matrix.length;
  const labels = regimeNames ?? Array.from({ length: n }, (_, i) => `状态 ${i}`);

  // Format values as text annotations
  const text = matrix.map(row => row.map(v => fmt(v, 3)));

  const traces: Data[] = [
    {
      z: matrix,
      x: labels,
      y: labels,
      type: "heatmap" as const,
      colorscale: [
        [0, "oklch(0.15 0.02 260)"],
        [0.5, "oklch(0.4 0.1 250)"],
        [1, "oklch(0.7 0.15 250)"],
      ],
      text,
      texttemplate: "%{text}",
      textfont: { size: 14, color: "white" },
      hovertemplate: "%{y} → %{x}: %{z:.3f}<extra></extra>",
      showscale: true,
      colorbar: {
        title: { text: "概率" },
        tickvals: [0, 0.5, 1],
      },
    } as unknown as Data,
  ];

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: { title: { text: "转移到" }, side: "bottom" as const },
        yaxis: { title: { text: "当前状态" }, autorange: "reversed" as const },
        margin: { l: 80, r: 30, t: 30, b: 60 },
      }}
      height={height}
      ariaLabel="转移概率矩阵热力图"
    />
  );
}
