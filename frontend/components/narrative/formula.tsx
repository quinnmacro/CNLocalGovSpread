"use client";

import { BlockMath, InlineMath } from "react-katex";
import "katex/dist/katex.min.css";

interface FormulaProps {
  math: string;
  block?: boolean;
  className?: string;
}

/**
 * Renders a KaTeX formula. Use block=true for display math.
 */
export function Formula({ math, block = false, className }: FormulaProps) {
  if (block) {
    return (
      <div className={className}>
        <BlockMath math={math} />
      </div>
    );
  }
  return <InlineMath math={math} />;
}
