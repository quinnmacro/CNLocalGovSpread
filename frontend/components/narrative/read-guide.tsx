"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";

interface ReadGuideProps {
  children: React.ReactNode;
  defaultOpen?: boolean;
  className?: string;
}

/**
 * Collapsible "📖 读图指南" component.
 * Provides context and reading instructions for charts.
 */
export function ReadGuide({
  children,
  defaultOpen = false,
  className,
}: ReadGuideProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <CollapsibleTrigger
        className={cn(
          "flex items-center gap-2 text-sm font-medium text-primary hover:text-primary/80 transition-colors py-1",
          className
        )}
      >
        <span>📖 读图指南</span>
        <ChevronDown
          className={cn(
            "h-4 w-4 transition-transform duration-200",
            isOpen && "rotate-180"
          )}
        />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="mt-2 pl-4 border-l-2 border-primary/30 text-sm text-muted-foreground space-y-1">
          {children}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
