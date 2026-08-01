"use client";

import { useState } from "react";
import { ChevronDown, BookOpen } from "lucide-react";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
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
 * Collapsible "读图指南" component with pill trigger + smooth expand.
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
          "inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium",
          "bg-primary/10 text-primary hover:bg-primary/15 border border-primary/20",
          "transition-all duration-200 cursor-pointer",
          className,
        )}
      >
        <BookOpen className="h-3.5 w-3.5" />
        <span>读图指南</span>
        <motion.div
          animate={{ rotate: isOpen ? 180 : 0 }}
          transition={{ duration: 0.25, ease: "easeInOut" }}
        >
          <ChevronDown className="h-3.5 w-3.5" />
        </motion.div>
      </CollapsibleTrigger>
      <AnimatePresence initial={false}>
        {isOpen && (
          <CollapsibleContent>
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
              className="overflow-hidden"
            >
              <div className="mt-3 pl-4 py-3 border-l-2 border-primary/20 bg-primary/[0.03] rounded-r-lg text-sm text-muted-foreground space-y-2 leading-relaxed">
                {children}
              </div>
            </motion.div>
          </CollapsibleContent>
        )}
      </AnimatePresence>
    </Collapsible>
  );
}
