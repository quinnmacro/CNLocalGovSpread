import { cn } from "@/lib/utils";

interface ProseBlockProps {
  className?: string;
  children: React.ReactNode;
}

/**
 * Styled prose block for narrative paragraphs.
 * Uses the prose-narrative utility class from globals.css.
 */
export function ProseBlock({ className, children }: ProseBlockProps) {
  return (
    <div className={cn("prose-narrative", className)}>
      {children}
    </div>
  );
}
