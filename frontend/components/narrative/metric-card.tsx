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
  /** Accent color for top border: "blue" | "green" | "orange" | "pink" | "cyan" */
  accent?: "blue" | "green" | "orange" | "pink" | "cyan";
}

const accentMap = {
  blue: "accent-border-blue",
  green: "accent-border-green",
  orange: "accent-border-orange",
  pink: "accent-border-pink",
  cyan: "accent-border-cyan",
} as const;

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
  const h = 28;
  const w = 80;
  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = h - ((v - min) / range) * h;
      return `${x},${y}`;
    })
    .join(" ");

  // Gradient fill area
  const areaPoints = `0,${h} ${points} ${w},${h}`;

  return (
    <svg
      width={w}
      height={h}
      className="text-primary/60"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="sparkGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="currentColor" stopOpacity="0.2" />
          <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon
        fill="url(#sparkGrad)"
        points={areaPoints}
      />
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
 * Metric display card with institutional aesthetic:
 * gradient top accent, hover glow, mono typography.
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
  accent,
}: MetricCardProps) {
  const displayValue =
    typeof value === "string"
      ? value
      : value != null
      ? fmt(value, decimals)
      : "—";

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
    >
      <Card
        className={cn(
          "metric-card glow-hover p-4 relative overflow-hidden",
          accent && accentMap[accent],
          className,
        )}
      >
        <div className="flex items-start justify-between mb-2">
          <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-widest leading-none">
            {label}
          </span>
          {icon && (
            <span className="text-muted-foreground/70 shrink-0">{icon}</span>
          )}
        </div>
        <div className="flex items-baseline gap-1.5 mb-1">
          <span className="text-2xl font-bold font-mono text-foreground tracking-tight">
            {displayValue}
          </span>
          {unit && (
            <span className="text-xs text-muted-foreground font-medium">
              {unit}
            </span>
          )}
        </div>
        <div className="flex items-center justify-between mt-2.5">
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
                    : "text-muted-foreground",
                )}
              >
                {change > 0 ? "+" : ""}
                {fmt(change, decimals)}
              </span>
            )}
            {changeLabel && (
              <span className="text-[11px] text-muted-foreground ml-0.5">
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
