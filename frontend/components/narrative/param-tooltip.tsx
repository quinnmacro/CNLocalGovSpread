"use client";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { fmt } from "@/lib/utils";

interface ParamTooltipProps {
  name: string;
  value: number | string | null;
  decimals?: number;
  tooltip: React.ReactNode;
  className?: string;
}

/**
 * Displays a parameter value with an economics tooltip on hover.
 * Example: α+β = 0.96 → "持续性参数，冲击衰减约17天"
 */
export function ParamTooltip({
  name,
  value,
  decimals = 4,
  tooltip,
  className,
}: ParamTooltipProps) {
  const displayValue =
    typeof value === "string" ? value : value != null ? fmt(value, decimals) : "—";

  return (
    <Tooltip>
      <TooltipTrigger
        className={cn(
          "inline-flex items-center gap-1.5 cursor-help border-b border-dashed border-muted-foreground/40 hover:border-primary/60 transition-colors",
          className
        )}
      >
        <span className="text-muted-foreground text-sm">{name}</span>
        <span className="font-mono font-medium text-foreground">
          {displayValue}
        </span>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-xs text-sm leading-relaxed">
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}
