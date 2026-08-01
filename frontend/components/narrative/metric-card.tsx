"use client";

import { cn, fmt } from "@/lib/utils";
import { Card } from "@/components/ui/card";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { motion } from "framer-motion";

interface MetricCardProps {
  label: string;
  value: number | string | null;
  unit?: string;
  decimals?: number;
  change?: number | null;
  changeLabel?: string;
  sparkline?: number[];
  className?: string;
  icon?: React.ReactNode;
}

function TrendIcon({ change }: { change?: number | null }) {
  if (change == null || change === 0)
    return <Minus className="h-3.5 w-3.5 text-muted-foreground" />;
  if (change > 0)
    return <TrendingUp className="h-3.5 w-3.5 text-chart-3" />;
  return <TrendingDown className="h-3.5 w-3.5 text-chart-2" />;
}

function MiniSparkline({ data }: { data: number[] }) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const h = 24;
  const w = 60;
  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = h - ((v - min) / range) * h;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg
      width={w}
      height={h}
      className="text-primary/60"
      aria-hidden="true"
    >
      <polyline
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        points={points}
      />
    </svg>
  );
}

/**
 * Metric display card: value + trend + optional sparkline.
 */
export function MetricCard({
  label,
  value,
  unit,
  decimals = 2,
  change,
  changeLabel,
  sparkline,
  className,
  icon,
}: MetricCardProps) {
  const displayValue =
    typeof value === "string"
      ? value
      : value != null
      ? fmt(value, decimals)
      : "—";

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <Card className={cn("metric-card p-4", className)}>
        <div className="flex items-start justify-between mb-1">
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            {label}
          </span>
          {icon && <span className="text-muted-foreground">{icon}</span>}
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold font-mono text-foreground">
            {displayValue}
          </span>
          {unit && (
            <span className="text-sm text-muted-foreground">{unit}</span>
          )}
        </div>
        <div className="flex items-center justify-between mt-2">
          <div className="flex items-center gap-1.5">
            <TrendIcon change={change} />
            {change != null && (
              <span
                className={cn(
                  "text-xs font-mono",
                  change > 0
                    ? "text-chart-3"
                    : change < 0
                    ? "text-chart-2"
                    : "text-muted-foreground"
                )}
              >
                {change > 0 ? "+" : ""}
                {fmt(change, decimals)}
              </span>
            )}
            {changeLabel && (
              <span className="text-xs text-muted-foreground ml-1">
                {changeLabel}
              </span>
            )}
          </div>
          {sparkline && sparkline.length > 1 && (
            <MiniSparkline data={sparkline} />
          )}
        </div>
      </Card>
    </motion.div>
  );
}
