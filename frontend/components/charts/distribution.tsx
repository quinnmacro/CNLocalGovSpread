"use client";
import { PlotlyChart, chartColors } from "./plotly-chart";
import type { Data } from "plotly.js";

interface DistributionChartProps {
  series: { name: string; values: number[]; color?: string }[];
  height?: number;
}

// Simple KDE using Gaussian kernel
function gaussianKde(data: number[], points: number[], bandwidth?: number): number[] {
  const h = bandwidth ?? (1.06 * std(data) * Math.pow(data.length, -0.2));
  if (h === 0) return points.map(() => 0);
  return points.map(x => {
    const sum = data.reduce((acc, xi) => {
      const z = (x - xi) / h;
      return acc + Math.exp(-0.5 * z * z);
    }, 0);
    return sum / (data.length * h * Math.sqrt(2 * Math.PI));
  });
}

function std(arr: number[]): number {
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  return Math.sqrt(arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length);
}

export function DistributionChart({ series, height = 400 }: DistributionChartProps) {
  const traces: Data[] = [];

  series.forEach((s, i) => {
    const color = s.color ?? chartColors[i % chartColors.length];
    const min = Math.min(...s.values);
    const max = Math.max(...s.values);
    const range = max - min;
    const xPts = Array.from({ length: 200 }, (_, j) => min - range * 0.1 + (range * 1.2 * j) / 199);
    const kde = gaussianKde(s.values, xPts);

    // Histogram
    traces.push({
      x: s.values,
      type: "histogram" as const,
      name: `${s.name} (直方图)`,
      histnorm: "probability density" as const,
      opacity: 0.3,
      marker: { color },
      nbinsx: 50,
      showlegend: false,
    });

    // KDE line
    traces.push({
      x: xPts,
      y: kde,
      type: "scatter" as const,
      mode: "lines" as const,
      name: s.name,
      line: { color, width: 2 },
    });

    // Normal reference
    const mu = s.values.reduce((a, b) => a + b, 0) / s.values.length;
    const sigma = std(s.values);
    const normalY = xPts.map(x => Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI)));
    traces.push({
      x: xPts,
      y: normalY,
      type: "scatter" as const,
      mode: "lines" as const,
      name: `${s.name} (正态参考)`,
      line: { color, width: 1, dash: "dash" as const },
      opacity: 0.5,
    });
  });

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: { title: { text: "利差 (bps)" } },
        yaxis: { title: { text: "概率密度" } },
        barmode: "overlay" as const,
        legend: { orientation: "h", y: -0.2 },
        hovermode: "x unified",
      }}
      height={height}
      ariaLabel="利差分布图"
    />
  );
}
