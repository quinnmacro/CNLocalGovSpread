"use client";
import { PlotlyChart, chartColors } from "./plotly-chart";
import type { TimePoint } from "@/lib/types";
import type { Data } from "plotly.js";

interface HARRVDecompositionProps {
  rvDaily: TimePoint[];
  rvWeekly: TimePoint[];
  rvMonthly: TimePoint[];
  condVol?: TimePoint[];
  height?: number;
}

export function HARRVDecomposition({
  rvDaily,
  rvWeekly,
  rvMonthly,
  condVol,
  height = 420,
}: HARRVDecompositionProps) {
  const traces: Data[] = [
    {
      x: rvDaily.map((p) => p.date),
      y: rvDaily.map((p) => p.value),
      type: "scatter" as const,
      mode: "lines" as const,
      name: "RV 日 (1d)",
      line: { color: chartColors[0], width: 1 },
      opacity: 0.5,
    },
    {
      x: rvWeekly.map((p) => p.date),
      y: rvWeekly.map((p) => p.value),
      type: "scatter" as const,
      mode: "lines" as const,
      name: "RV 周 (5d)",
      line: { color: chartColors[1], width: 1.5 },
    },
    {
      x: rvMonthly.map((p) => p.date),
      y: rvMonthly.map((p) => p.value),
      type: "scatter" as const,
      mode: "lines" as const,
      name: "RV 月 (22d)",
      line: { color: chartColors[2], width: 2 },
    },
  ];

  if (condVol && condVol.length > 0) {
    traces.push({
      x: condVol.map((p) => p.date),
      y: condVol.map((p) => p.value ** 2),
      type: "scatter" as const,
      mode: "lines" as const,
      name: "HAR 拟合 σ²",
      line: { color: chartColors[3], width: 2, dash: "dash" as const },
    });
  }

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: { title: { text: "日期" } },
        yaxis: { title: { text: "已实现方差" } },
        legend: { orientation: "h" as const, y: -0.2 },
        hovermode: "x unified" as const,
      }}
      height={height}
      ariaLabel="HAR-RV 三窗口分解图"
    />
  );
}
