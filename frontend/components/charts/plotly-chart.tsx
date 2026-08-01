"use client";

import dynamic from "next/dynamic";
import { cn } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";
import type { Data, Layout, Config } from "plotly.js";

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => (
    <Skeleton className="w-full h-80 rounded-lg" />
  ),
});

/** Dark theme defaults for all Plotly charts */
const darkLayout: Partial<Layout> = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: {
    color: "oklch(0.65 0.01 260)",
    family: "Inter, system-ui, sans-serif",
    size: 12,
  },
  xaxis: {
    gridcolor: "oklch(0.22 0.015 260)",
    zerolinecolor: "oklch(0.28 0.02 260)",
    linecolor: "oklch(0.28 0.02 260)",
    tickfont: { size: 10 },
    title: { font: { size: 11 } },
  },
  yaxis: {
    gridcolor: "oklch(0.22 0.015 260)",
    zerolinecolor: "oklch(0.28 0.02 260)",
    linecolor: "oklch(0.28 0.02 260)",
    tickfont: { size: 10 },
    title: { font: { size: 11 } },
  },
  legend: {
    bgcolor: "rgba(0,0,0,0)",
    font: { color: "oklch(0.65 0.01 260)", size: 11 },
  },
  margin: { l: 50, r: 30, t: 30, b: 40 },
  autosize: true,
  hoverlabel: {
    bgcolor: "oklch(0.19 0.015 260)",
    bordercolor: "oklch(0.28 0.02 260)",
    font: { color: "oklch(0.95 0.005 260)", size: 12, family: "Inter, system-ui" },
  },
  colorway: [
    "oklch(0.75 0.15 250)",
    "oklch(0.70 0.15 170)",
    "oklch(0.75 0.15 50)",
    "oklch(0.70 0.15 330)",
    "oklch(0.75 0.15 90)",
    "oklch(0.70 0.15 210)",
    "oklch(0.65 0.15 20)",
  ],
};

const defaultConfig: Partial<Config> = {
  responsive: true,
  displayModeBar: false,
  scrollZoom: true,
};

/** Institutional color palette for chart series */
export const chartColors = [
  "oklch(0.75 0.15 250)", // blue
  "oklch(0.70 0.15 170)", // teal
  "oklch(0.75 0.15 50)",  // orange
  "oklch(0.70 0.15 330)", // pink
  "oklch(0.75 0.15 90)",  // yellow-green
  "oklch(0.70 0.15 210)", // cyan
  "oklch(0.65 0.15 20)",  // red
] as const;

interface PlotlyChartProps {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  className?: string;
  height?: number;
  ariaLabel?: string;
}

export function PlotlyChart({
  data,
  layout,
  config,
  className,
  height = 400,
  ariaLabel = "数据图表",
}: PlotlyChartProps) {
  const mergedLayout: Partial<Layout> = {
    ...darkLayout,
    ...layout,
    height,
    xaxis: { ...darkLayout.xaxis, ...layout?.xaxis },
    yaxis: { ...darkLayout.yaxis, ...layout?.yaxis },
    margin: { ...darkLayout.margin, ...layout?.margin },
    hoverlabel: { ...darkLayout.hoverlabel, ...layout?.hoverlabel },
  };

  return (
    <div
      className={cn("chart-container w-full", className)}
      role="img"
      aria-label={ariaLabel}
    >
      <Plot
        data={data}
        layout={mergedLayout as Layout}
        config={{ ...defaultConfig, ...config } as Config}
        style={{ width: "100%", height }}
        useResizeHandler
      />
    </div>
  );
}
