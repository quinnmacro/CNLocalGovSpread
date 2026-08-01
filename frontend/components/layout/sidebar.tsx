"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";

const analysisLinks = [
  {
    href: "/analysis/overview",
    label: "利差全景",
    labelEn: "Overview",
    emoji: "📊",
  },
  {
    href: "/analysis/volatility",
    label: "波动率建模",
    labelEn: "Volatility",
    emoji: "📈",
  },
  {
    href: "/analysis/risk",
    label: "风险度量",
    labelEn: "Risk",
    emoji: "⚠️",
  },
  {
    href: "/analysis/regimes",
    label: "市场状态",
    labelEn: "Regimes",
    emoji: "🔄",
  },
  {
    href: "/analysis/scenarios",
    label: "情景分析",
    labelEn: "Scenarios",
    emoji: "🔮",
  },
];

export function Sidebar() {
  const pathname = usePathname();
  const isAnalysisPage = pathname.startsWith("/analysis/");

  if (!isAnalysisPage) return null;

  return (
    <aside className="hidden lg:block w-56 shrink-0 border-r border-border/40">
      <ScrollArea className="h-[calc(100vh-3.5rem)] py-4 px-3">
        <div className="mb-4 px-3">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            分析模块
          </h2>
        </div>
        <nav className="flex flex-col gap-1">
          {analysisLinks.map((link, i) => {
            const isActive = pathname === link.href;
            return (
              <div key={link.href}>
                <Link
                  href={link.href}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2 text-sm rounded-md transition-colors",
                    isActive
                      ? "bg-primary/10 text-primary font-medium"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted"
                  )}
                >
                  <span className="text-base">{link.emoji}</span>
                  <div>
                    <div className="leading-tight">{link.label}</div>
                    <div className="text-[10px] font-mono text-muted-foreground/60">
                      {link.labelEn}
                    </div>
                  </div>
                </Link>
                {i < analysisLinks.length - 1 && (
                  <Separator className="my-1 opacity-30" />
                )}
              </div>
            );
          })}
        </nav>
      </ScrollArea>
    </aside>
  );
}
