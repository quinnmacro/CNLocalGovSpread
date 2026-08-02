"use client";
import { PlotlyChart, chartColors } from "./plotly-chart";
import type { TimePoint } from "@/lib/types";
import type { Data } from "plotly.js";

interface MSGARCHRegimesProps {
  condVol: TimePoint[];
  regimeLabels: TimePoint[];
  regimeParams: {
    regime: number;
    omega: number;
    alpha: number;
    beta: number;
    persistence: number;
    mean_abs_return: number;
  }[];
  height?: number;
}

const regimeColors = ["#22c55e", "#f59e0b", "#ef4444", "#8b5cf6"];
const regimeNames = ["低波动", "高波动", "极端波动", "危机"];

export function MSGARCHRegimes({
  condVol,
  regimeLabels,
  regimeParams,
  height = 450,
}: MSGARCHRegimesProps) {
  const nRegimes = regimeParams.length;
  const traces: Data[] = [];

  // Plot volatility colored by regime
  for (let k = 0; k < nRegimes; k++) {
    const regimeDates = condVol
      .filter((_, i) => Math.round(regimeLabels[i]?.value ?? -1) === k)
      .map((p) => p.date);
    const regimeVols = condVol
      .filter((_, i) => Math.round(regimeLabels[i]?.value ?? -1) === k)
      .map((p) => p.value);

    traces.push({
      x: regimeDates,
      y: regimeVols,
      type: "scatter" as const,
      mode: "markers" as const,
      name: regimeNames[k] ?? `Regime ${k}`,
      marker: {
        color: regimeColors[k % regimeColors.length],
        size: 3,
        opacity: 0.7,
      },
    });
  }

  // Full volatility line (light)
  traces.push({
    x: condVol.map((p) => p.date),
    y: condVol.map((p) => p.value),
    type: "scatter" as const,
    mode: "lines" as const,
    name: "MS-GARCH σ",
    line: { color: "rgba(150,150,150,0.5)", width: 1 },
  });

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
      ariaLabel="MS-GARCH 状态切换波动率图"
    />
  );
}
