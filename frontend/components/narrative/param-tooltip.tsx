"use client";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { fmt } from "@/lib/utils";
import { Info } from "lucide-react";

interface ParamTooltipProps {
  name: string;
  value: number | string | null;
  decimals?: number;
  tooltip: React.ReactNode;
  className?: string;
}

/**
 * Displays a parameter value with info icon + economics tooltip on hover.
 * Glassmorphism-style tooltip background.
 */
export function ParamTooltip({
  name,
  value,
  decimals = 4,
  tooltip,
  className,
}: ParamTooltipProps) {
  const displayValue =
    typeof value === "string"
      ? value
      : value != null
      ? fmt(value, decimals)
      : "—";

  return (
    <Tooltip>
      <TooltipTrigger
        className={cn(
          "inline-flex items-center gap-1.5 cursor-help",
          "px-2 py-0.5 rounded-md bg-muted/50 hover:bg-muted/80",
          "border border-border/50 hover:border-primary/40",
          "transition-all duration-200 group",
          className,
        )}
      >
        <Info className="h-3 w-3 text-muted-foreground/60 group-hover:text-primary/60 transition-colors" />
        <span className="text-muted-foreground text-xs">{name}</span>
        <span className="font-mono font-semibold text-foreground text-xs">
          {displayValue}
        </span>
      </TooltipTrigger>
      <TooltipContent
        side="top"
        sideOffset={8}
        className="max-w-xs text-sm leading-relaxed bg-popover/95 backdrop-blur-md border-border/50 shadow-xl"
      >
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}
