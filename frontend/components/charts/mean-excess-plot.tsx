"use client";
import { PlotlyChart, chartColors } from "./plotly-chart";
import type { MeanExcessPoint } from "@/lib/types";
import type { Data } from "plotly.js";

interface MeanExcessPlotProps {
  points: MeanExcessPoint[];
  height?: number;
}

export function MeanExcessPlot({ points, height = 350 }: MeanExcessPlotProps) {
  const thresholds = points.map(p => p.threshold);
  const excesses = points.map(p => p.mean_excess);

  // Linear fit for the tail region (last 40%)
  const tailStart = Math.floor(points.length * 0.6);
  const tailX = thresholds.slice(tailStart);
  const tailY = excesses.slice(tailStart);

  // Simple linear regression
  const n = tailX.length;
  const sumX = tailX.reduce((a, b) => a + b, 0);
  const sumY = tailY.reduce((a, b) => a + b, 0);
  const sumXY = tailX.reduce((a, x, i) => a + x * tailY[i], 0);
  const sumX2 = tailX.reduce((a, x) => a + x * x, 0);
  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;

  const fitLine = tailX.map(x => slope * x + intercept);

  const traces: Data[] = [
    {
      x: thresholds,
      y: excesses,
      type: "scatter" as const,
      mode: "markers" as const,
      name: "均值超额",
      marker: { color: chartColors[0], size: 4, opacity: 0.7 },
    },
    {
      x: tailX,
      y: fitLine,
      type: "scatter" as const,
      mode: "lines" as const,
      name: "线性拟合 (尾部)",
      line: { color: chartColors[2], dash: "dash" as const, width: 1.5 },
    },
  ];

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: { title: { text: "阈值 u" } },
        yaxis: { title: { text: "均值超额 E(X-u | X>u)" } },
        legend: { orientation: "h", y: -0.2 },
        hovermode: "closest",
      }}
      height={height}
      ariaLabel="均值超额图"
    />
  );
}
