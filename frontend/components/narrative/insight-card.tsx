"use client";

import { cn } from "@/lib/utils";
import { Lightbulb } from "lucide-react";
import { motion } from "framer-motion";

interface InsightCardProps {
  title?: string;
  children: React.ReactNode;
  variant?: "info" | "warning" | "success";
  className?: string;
}

const variantStyles = {
  info: "border-primary/30 bg-primary/5",
  warning: "border-chart-3/30 bg-chart-3/5",
  success: "border-chart-2/30 bg-chart-2/5",
} as const;

const iconColors = {
  info: "text-primary",
  warning: "text-chart-3",
  success: "text-chart-2",
} as const;

/**
 * Highlighted "发现" / insight callout card.
 * Used to emphasize key findings in the narrative.
 */
export function InsightCard({
  title = "发现",
  children,
  variant = "info",
  className,
}: InsightCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4 }}
      className={cn(
        "rounded-lg border-l-4 p-4 my-6",
        variantStyles[variant],
        className
      )}
    >
      <div className="flex items-center gap-2 mb-2">
        <Lightbulb className={cn("h-4 w-4", iconColors[variant])} />
        <span className="text-sm font-semibold text-foreground">{title}</span>
      </div>
      <div className="text-sm text-muted-foreground leading-relaxed">
        {children}
      </div>
    </motion.div>
  );
}
