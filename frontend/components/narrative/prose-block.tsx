import { cn } from "@/lib/utils";

interface ProseBlockProps {
  className?: string;
  children: React.ReactNode;
  /** "default" for standard prose, "callout" for emphasized block with left accent */
  variant?: "default" | "callout";
}

/**
 * Styled prose block for narrative paragraphs.
 * Uses the prose-narrative utility class from globals.css.
 * Optional callout variant adds left border + subtle background.
 */
export function ProseBlock({
  className,
  children,
  variant = "default",
}: ProseBlockProps) {
  if (variant === "callout") {
    return (
      <div
        className={cn(
          "prose-narrative my-4 pl-4 py-3 border-l-2 border-primary/25",
          "bg-primary/[0.03] rounded-r-lg",
          className,
        )}
      >
        {children}
      </div>
    );
  }
  return (
    <div className={cn("prose-narrative", className)}>{children}</div>
  );
}
