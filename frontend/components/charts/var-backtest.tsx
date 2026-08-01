"use client";
import { PlotlyChart, chartColors } from "./plotly-chart";
import type { TimePoint, KupiecTest, ChristoffersenTest } from "@/lib/types";
import type { Data } from "plotly.js";
import { fmt } from "@/lib/utils";

interface VarBacktestChartProps {
  varSeries: TimePoint[];
  returns?: TimePoint[];
  violations: number;
  nObservations: number;
  actualCoverage: number;
  kupiec: KupiecTest;
  christoffersen: ChristoffersenTest;
  height?: number;
}

export function VarBacktestChart({
  varSeries,
  returns,
  violations,
  nObservations,
  actualCoverage,
  kupiec,
  christoffersen,
  height = 400,
}: VarBacktestChartProps) {
  const traces: Data[] = [];

  // VaR line
  traces.push({
    x: varSeries.map(v => v.date),
    y: varSeries.map(v => v.value),
    type: "scatter" as const,
    mode: "lines" as const,
    name: "VaR",
    line: { color: chartColors[2], width: 1.5 },
  });

  // Returns if available
  if (returns && returns.length > 0) {
    traces.push({
      x: returns.map(r => r.date),
      y: returns.map(r => r.value),
      type: "scatter" as const,
      mode: "lines" as const,
      name: "日收益率",
      line: { color: "oklch(0.5 0.01 260)", width: 0.8 },
      opacity: 0.6,
    });

    // Mark violations
    const varMap = new Map(varSeries.map(v => [v.date, v.value]));
    const violationDates: string[] = [];
    const violationValues: number[] = [];
    for (const r of returns) {
      const varVal = varMap.get(r.date);
      if (varVal != null && r.value < -Math.abs(varVal)) {
        violationDates.push(r.date);
        violationValues.push(r.value);
      }
    }
    if (violationDates.length > 0) {
      traces.push({
        x: violationDates,
        y: violationValues,
        type: "scatter" as const,
        mode: "markers" as const,
        name: `违规 (${violations}次)`,
        marker: { color: "oklch(0.6 0.2 25)", size: 5, symbol: "x" as const },
      });
    }
  }

  return (
    <PlotlyChart
      data={traces}
      layout={{
        xaxis: { title: { text: "日期" } },
        yaxis: { title: { text: "收益率 / VaR (bps)" } },
        legend: { orientation: "h", y: -0.2 },
        hovermode: "x unified",
        annotations: [
          {
            x: 0.02, y: 0.95,
            xref: "paper" as const, yref: "paper" as const,
            text: `覆盖率: ${fmt(actualCoverage * 100, 2)}% | Kupiec p=${fmt(kupiec.pvalue, 4)} | Christoffersen p=${fmt(christoffersen.pvalue, 4)}`,
            showarrow: false,
            font: { size: 10, color: "oklch(0.65 0.01 260)" },
            align: "left" as const,
          },
        ],
      }}
      height={height}
      ariaLabel="VaR 回测时序图"
    />
  );
}
