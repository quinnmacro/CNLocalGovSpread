"use client";

import { cn } from "@/lib/utils";
import { Lightbulb, AlertTriangle, CheckCircle, Info } from "lucide-react";
import { motion } from "framer-motion";

interface InsightCardProps {
  title?: string;
  children: React.ReactNode;
  variant?: "info" | "warning" | "success";
  className?: string;
}

const variantConfig = {
  info: {
    icon: Info,
    glow: "insight-glow-info",
    border: "border-primary/30",
    bg: "bg-gradient-to-r from-primary/[0.06] to-transparent",
    iconBg: "bg-primary/15",
    iconColor: "text-primary",
    titleColor: "text-primary",
  },
  warning: {
    icon: AlertTriangle,
    glow: "insight-glow-warning",
    border: "border-chart-3/30",
    bg: "bg-gradient-to-r from-chart-3/[0.06] to-transparent",
    iconBg: "bg-chart-3/15",
    iconColor: "text-chart-3",
    titleColor: "text-chart-3",
  },
  success: {
    icon: CheckCircle,
    glow: "insight-glow-success",
    border: "border-chart-2/30",
    bg: "bg-gradient-to-r from-chart-2/[0.06] to-transparent",
    iconBg: "bg-chart-2/15",
    iconColor: "text-chart-2",
    titleColor: "text-chart-2",
  },
} as const;

/**
 * Highlighted insight callout card with variant-specific glow + gradient.
 */
export function InsightCard({
  title = "发现",
  children,
  variant = "info",
  className,
}: InsightCardProps) {
  const cfg = variantConfig[variant];
  const Icon = cfg.icon;

  return (
    <motion.div
      initial={{ opacity: 0, x: -12 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.45, ease: "easeOut" }}
      className={cn(
        "rounded-xl border-l-[3px] p-5 my-6 relative overflow-hidden",
        cfg.border,
        cfg.bg,
        cfg.glow,
        className,
      )}
    >
      <div className="flex items-center gap-2.5 mb-3">
        <span
          className={cn(
            "flex items-center justify-center h-6 w-6 rounded-full shrink-0",
            cfg.iconBg,
          )}
        >
          <Icon className={cn("h-3.5 w-3.5", cfg.iconColor)} />
        </span>
        <span className={cn("text-sm font-semibold", cfg.titleColor)}>
          {title}
        </span>
      </div>
      <div className="text-sm text-muted-foreground leading-relaxed pl-[34px]">
        {children}
      </div>
    </motion.div>
  );
}
