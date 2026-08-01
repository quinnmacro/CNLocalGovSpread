"use client";

import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface SectionProps {
  id?: string;
  index?: number;
  title: string;
  subtitle?: string;
  className?: string;
  children: React.ReactNode;
}

const segmentLabels = [
  "WHY — 研究动机",
  "HOW — 方法论",
  "WHAT — 结果展示",
  "SO WHAT — 诊断与解读",
  "NOW WHAT — 投资含义",
] as const;

export function Section({
  id,
  index,
  title,
  subtitle,
  className,
  children,
}: SectionProps) {
  return (
    <motion.section
      id={id}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className={cn("py-12 md:py-16", className)}
    >
      <div className="max-w-4xl mx-auto">
        {index != null && index >= 0 && index < segmentLabels.length && (
          <div className="flex items-center gap-3 mb-2">
            <span className="text-xs font-mono uppercase tracking-wider text-primary/70 bg-primary/10 px-2 py-0.5 rounded">
              {segmentLabels[index]}
            </span>
          </div>
        )}
        <h2 className="text-2xl md:text-3xl font-bold tracking-tight text-foreground mb-2">
          {title}
        </h2>
        {subtitle && (
          <p className="text-muted-foreground text-lg mb-8">{subtitle}</p>
        )}
        <div className="prose-narrative">{children}</div>
      </div>
    </motion.section>
  );
}
