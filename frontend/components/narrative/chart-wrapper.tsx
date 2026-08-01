"use client";

import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface ChartMetric {
  label: string;
  value: string;
  color?: string;
}

interface ChartWrapperProps {
  title: string;
  subtitle?: string;
  metrics?: ChartMetric[];
  icon?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

/**
 * Consistent chart container with header bar, metric strip, and content area.
 * Provides the Bloomberg-terminal aesthetic for all charts.
 */
export function ChartWrapper({
  title,
  subtitle,
  metrics,
  icon,
  children,
  className,
}: ChartWrapperProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className={cn(
        "rounded-xl border border-border/60 bg-card/80 backdrop-blur-sm overflow-hidden",
        className,
      )}
    >
      {/* Header bar */}
      <div className="chart-header">
        <div className="flex items-center gap-2.5">
          {icon && (
            <span className="text-primary/70 shrink-0">{icon}</span>
          )}
          <div>
            <h3 className="text-sm font-semibold text-foreground tracking-tight">
              {title}
            </h3>
            {subtitle && (
              <p className="text-[11px] text-muted-foreground mt-0.5">
                {subtitle}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Metric strip */}
      {metrics && metrics.length > 0 && (
        <div className="metric-strip">
          {metrics.map((m) => (
            <div key={m.label} className="metric-item">
              <span className="metric-label">{m.label}:</span>
              <span
                className="metric-value"
                style={m.color ? { color: m.color } : undefined}
              >
                {m.value}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Chart content */}
      <div className="p-2">{children}</div>
    </motion.div>
  );
}
