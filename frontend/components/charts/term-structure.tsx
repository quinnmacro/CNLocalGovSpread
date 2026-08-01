"use client";
import { PlotlyChart, chartColors } from "./plotly-chart";
import type { TimePoint } from "@/lib/types";
import type { Data } from "plotly.js";

interface TermStructureChartProps {
  series5y: TimePoint[];
  series10y: TimePoint[];
  series30y: TimePoint[];
  height?: number;
}

export function TermStructureChart({ series5y, series10y, series30y, height = 400 }: TermStructureChartProps) {
  // Pair dates across maturities for scatter
  const map5y = new Map(series5y.map(p => [p.date, p.value]));
  const map10y = new Map(series10y.map(p => [p.date, p.value]));
  const map30y = new Map(series30y.map(p => [p.date, p.value]));

  const allDates = [...new Set([...map5y.keys(), ...map10y.keys()])];
  
  const x5y: number[] = [], y10y: number[] = [];
  const x5y30: number[] = [], y30y: number[] = [];

  for (const d of allDates) {
    const v5 = map5y.get(d);
    const v10 = map10y.get(d);
    const v30 = map30y.get(d);
    if (v5 != null && v10 != null) { x5y.push(v5); y10y.push(v10); }
    if (v5 != null && v30 != null) { x5y30.push(v5); y30y.push(v30); }
  }

  const traces: Data[] = [
    {
      x: x5y, y: y10y,
      type: "scatter" as const,
      mode: "markers" as const,
      name: "5Y vs 10Y",
      marker: { color: chartColors[0], size: 3, opacity: 0.4 },
    },
    {
      x: x5y30, y: y30y,
      type: "scatter" as const,
      mode: "markers" as const,
      name: "5Y vs 30Y",
      marker: { color: chartColors[2], size: 3, opacity: 0.4 },
    },
    // 45-degree reference line
    {
      x: [-5, 100], y: [-5, 100],
      type: "scatter" as const,
      mode: "lines" as const,
      name: "45° 参考线",
      line: { color: "oklch(0.4 0.01 260)", dash: "dash" as const, width: 1 },
      showlegend: true,
    },
  ];

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: { title: { text: "5Y 利差 (bps)" } },
        yaxis: { title: { text: "10Y / 30Y 利差 (bps)" } },
        legend: { orientation: "h", y: -0.15 },
        hovermode: "closest",
      }}
      height={height}
      ariaLabel="期限结构散点图"
    />
  );
}
