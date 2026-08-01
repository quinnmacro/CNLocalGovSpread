"use client";
import type { MarketGaugeResponse } from "@/lib/types";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { fmt } from "@/lib/utils";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { Activity, AlertTriangle, CheckCircle, Info } from "lucide-react";

interface MarketGaugePanelProps {
  gauge: MarketGaugeResponse;
}

const statusConfig: Record<
  string,
  {
    icon: typeof Activity;
    color: string;
    bgGlow: string;
    barColor: string;
    badgeBorder: string;
  }
> = {
  calm: {
    icon: CheckCircle,
    color: "text-chart-2",
    bgGlow: "shadow-[inset_0_0_0_1px_oklch(0.7_0.15_170/0.15),0_0_20px_oklch(0.7_0.15_170/0.08)]",
    barColor: "bg-chart-2",
    badgeBorder: "border-chart-2/30",
  },
  caution: {
    icon: Info,
    color: "text-chart-3",
    bgGlow: "shadow-[inset_0_0_0_1px_oklch(0.75_0.15_50/0.15),0_0_20px_oklch(0.75_0.15_50/0.08)]",
    barColor: "bg-chart-3",
    badgeBorder: "border-chart-3/30",
  },
  stress: {
    icon: AlertTriangle,
    color: "text-chart-3",
    bgGlow: "shadow-[inset_0_0_0_1px_oklch(0.75_0.15_50/0.2),0_0_20px_oklch(0.75_0.15_50/0.1)]",
    barColor: "bg-chart-3",
    badgeBorder: "border-chart-3/40",
  },
  crisis: {
    icon: AlertTriangle,
    color: "text-destructive",
    bgGlow: "shadow-[inset_0_0_0_1px_oklch(0.6_0.2_25/0.25),0_0_24px_oklch(0.6_0.2_25/0.12)]",
    barColor: "bg-destructive",
    badgeBorder: "border-destructive/40",
  },
};

const indicatorLabels: Record<string, string> = {
  spread_level: "利差水平",
  spread_momentum: "利差动量",
  volatility: "波动率",
  trend: "趋势强度",
  regime: "状态评分",
};

/** Progress bar with gradient and glow */
function GaugeBar({
  value,
  max = 100,
  colorClass,
}: {
  value: number;
  max?: number;
  colorClass: string;
}) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div className="h-1.5 w-full rounded-full bg-muted/50 overflow-hidden">
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${pct}%` }}
        transition={{ duration: 0.8, ease: "easeOut", delay: 0.2 }}
        className={cn("h-full rounded-full", colorClass)}
      />
    </div>
  );
}

export function MarketGaugePanel({ gauge }: MarketGaugePanelProps) {
  const statusKey = gauge.status[0] ?? "caution";
  const config = statusConfig[statusKey] ?? statusConfig.caution;
  const StatusIcon = config.icon;

  // Determine composite gauge color for the big number
  const gaugeColor =
    gauge.composite < 30
      ? "text-chart-2"
      : gauge.composite < 60
      ? "text-chart-3"
      : gauge.composite < 80
      ? "text-chart-3"
      : "text-destructive";

  return (
    <div className="space-y-6">
      {/* Main gauge display — glass card with glow */}
      <motion.div
        initial={{ opacity: 0, scale: 0.96 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
      >
        <Card
          className={cn(
            "p-6 text-center glass-card rounded-xl relative overflow-hidden",
            config.bgGlow,
          )}
        >
          {/* Subtle shimmer overlay */}
          <div className="absolute inset-0 shimmer-bg pointer-events-none" />

          <div className="relative z-10">
            <div className="flex items-center justify-center gap-3 mb-4">
              <StatusIcon className={cn("h-5 w-5", config.color)} />
              <Badge
                variant="outline"
                className={cn(
                  "text-sm px-3 py-1 font-medium",
                  config.color,
                  "border",
                  config.badgeBorder,
                )}
              >
                {gauge.status[1]}
              </Badge>
            </div>
            <div
              className={cn(
                "text-5xl font-bold font-mono tracking-tight mb-1",
                gaugeColor,
              )}
            >
              {fmt(gauge.composite, 1)}
            </div>
            <div className="text-xs text-muted-foreground mb-4">
              综合评分 (0–100)
            </div>

            {/* Composite progress bar */}
            <GaugeBar value={gauge.composite} colorClass={config.barColor} />
          </div>
        </Card>
      </motion.div>

      {/* Sub-indicators — compact metric rows */}
      <div className="rounded-xl border border-border/50 bg-card/60 backdrop-blur-sm overflow-hidden">
        <div className="px-4 py-2.5 border-b border-border/40 bg-muted/20">
          <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-widest">
            子指标明细
          </span>
        </div>
        <div className="divide-y divide-border/30">
          {Object.entries(gauge.indicators).map(
            ([key, indicator], idx) => {
              const score = Math.min(100, Math.max(0, indicator.score));
              const barColor =
                score < 30
                  ? "bg-chart-2"
                  : score < 60
                  ? "bg-chart-3"
                  : "bg-destructive";
              return (
                <motion.div
                  key={key}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: 0.1 + idx * 0.08 }}
                  className="flex items-center gap-3 px-4 py-3 hover:bg-muted/20 transition-colors"
                >
                  <span className="text-sm font-medium text-foreground min-w-[5.5rem]">
                    {indicatorLabels[key] ?? key}
                  </span>
                  <div className="flex-1">
                    <GaugeBar value={score} colorClass={barColor} />
                  </div>
                  <span className="text-sm font-mono font-medium text-foreground w-10 text-right">
                    {fmt(indicator.score, 1)}
                  </span>
                </motion.div>
              );
            },
          )}
        </div>
      </div>
    </div>
  );
}
