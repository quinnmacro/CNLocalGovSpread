"use client";

import Link from "next/link";
import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/card";
import { ArrowRight } from "lucide-react";
import { motion } from "framer-motion";

interface NavigationCardProps {
  href: string;
  emoji: string;
  title: string;
  titleEn: string;
  description: string;
  className?: string;
}

/**
 * Navigation card for analysis module entry points.
 */
export function NavigationCard({
  href,
  emoji,
  title,
  titleEn,
  description,
  className,
}: NavigationCardProps) {
  return (
    <motion.div
      whileHover={{ y: -4, scale: 1.02 }}
      transition={{ type: "spring", stiffness: 400, damping: 25 }}
    >
      <Link href={href}>
        <Card
          className={cn(
            "group relative h-full p-6 cursor-pointer transition-all duration-300",
            "border-border/50 hover:border-primary/40 hover:shadow-lg hover:shadow-primary/5",
            className
          )}
        >
          <div className="text-3xl mb-3">{emoji}</div>
          <h3 className="text-lg font-bold text-foreground mb-1">{title}</h3>
          <p className="text-xs font-mono text-primary/70 mb-2">{titleEn}</p>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {description}
          </p>
          <div className="absolute bottom-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
            <ArrowRight className="h-5 w-5 text-primary" />
          </div>
        </Card>
      </Link>
    </motion.div>
  );
}
