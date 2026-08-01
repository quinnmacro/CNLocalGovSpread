"use client";

import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { BookOpen, Lightbulb, TrendingUp, Target, Rocket } from "lucide-react";

interface SectionProps {
  id?: string;
  index?: number;
  title: string;
  subtitle?: string;
  className?: string;
  children: React.ReactNode;
}

const segmentConfig = [
  { label: "WHY — 研究动机", icon: BookOpen, gradient: "from-blue-500/20 to-cyan-500/20" },
  { label: "HOW — 方法论", icon: Lightbulb, gradient: "from-purple-500/20 to-pink-500/20" },
  { label: "WHAT — 结果展示", icon: TrendingUp, gradient: "from-emerald-500/20 to-teal-500/20" },
  { label: "SO WHAT — 诊断与解读", icon: Target, gradient: "from-orange-500/20 to-red-500/20" },
  { label: "NOW WHAT — 投资含义", icon: Rocket, gradient: "from-indigo-500/20 to-violet-500/20" },
] as const;

export function Section({
  id,
  index,
  title,
  subtitle,
  className,
  children,
}: SectionProps) {
  const config = index != null && index >= 0 && index < segmentConfig.length
    ? segmentConfig[index]
    : null;
  const Icon = config?.icon;

  return (
    <motion.section
      id={id}
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className={cn("py-10 md:py-14", className)}
    >
      <div className="max-w-4xl mx-auto">
        {config && Icon && (
          <div className="mb-6">
            <div className={cn(
              "inline-flex items-center gap-3 px-4 py-2 rounded-lg bg-gradient-to-r backdrop-blur-sm",
              config.gradient
            )}>
              <Icon className="h-5 w-5 text-primary" />
              <span className="text-sm font-mono font-semibold uppercase tracking-wider text-primary">
                {config.label}
              </span>
            </div>
          </div>
        )}
        <h2 className="text-2xl md:text-3xl font-bold tracking-tight text-foreground mb-2 leading-tight">
          {title}
        </h2>
        {subtitle && (
          <p className="text-muted-foreground text-lg mb-8 leading-relaxed">{subtitle}</p>
        )}
        <div className="prose-narrative">{children}</div>
      </div>
    </motion.section>
  );
}
