"use client";

import Link from "next/link";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface PageNavigationProps {
  prev?: { href: string; label: string; emoji: string };
  next?: { href: string; label: string; emoji: string };
  className?: string;
}

export function PageNavigation({ prev, next, className }: PageNavigationProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className={cn(
        "flex items-center justify-between gap-4 pt-8 mt-8 border-t border-border/40",
        className,
      )}
    >
      {prev ? (
        <Link
          href={prev.href}
          className={cn(
            "group flex items-center gap-3 px-4 py-3 rounded-lg",
            "border border-border/40 bg-card/30 hover:bg-card/60 hover:border-primary/30",
            "transition-all duration-200 flex-1 max-w-[45%]",
          )}
        >
          <ArrowLeft className="h-4 w-4 text-muted-foreground group-hover:text-primary transition-colors shrink-0" />
          <div className="min-w-0">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-0.5">
              上一模块
            </div>
            <div className="text-sm font-medium text-foreground truncate flex items-center gap-1.5">
              <span>{prev.emoji}</span>
              <span>{prev.label}</span>
            </div>
          </div>
        </Link>
      ) : (
        <div className="flex-1" />
      )}

      {next ? (
        <Link
          href={next.href}
          className={cn(
            "group flex items-center gap-3 px-4 py-3 rounded-lg",
            "border border-border/40 bg-card/30 hover:bg-card/60 hover:border-primary/30",
            "transition-all duration-200 flex-1 max-w-[45%] justify-end text-right",
          )}
        >
          <div className="min-w-0">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-0.5">
              下一模块
            </div>
            <div className="text-sm font-medium text-foreground truncate flex items-center gap-1.5">
              <span>{next.label}</span>
              <span>{next.emoji}</span>
            </div>
          </div>
          <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:text-primary transition-colors shrink-0" />
        </Link>
      ) : (
        <div className="flex-1" />
      )}
    </motion.div>
  );
}
