"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Menu, BarChart3 } from "lucide-react";
import { useState } from "react";

const navLinks = [
  { href: "/", label: "首页", labelEn: "Home" },
  { href: "/analysis/overview", label: "概览", labelEn: "Overview" },
  { href: "/analysis/volatility", label: "波动率", labelEn: "Volatility" },
  { href: "/analysis/risk", label: "风险", labelEn: "Risk" },
  { href: "/analysis/regimes", label: "状态", labelEn: "Regimes" },
  { href: "/analysis/scenarios", label: "情景", labelEn: "Scenarios" },
];

export function Navbar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/80 backdrop-blur-xl">
      <nav className="flex h-14 items-center px-4 md:px-6 max-w-7xl mx-auto">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2 mr-8">
          <BarChart3 className="h-5 w-5 text-primary" />
          <span className="font-bold text-sm tracking-tight text-foreground hidden sm:inline">
            QuinnMacro
          </span>
        </Link>

        {/* Desktop nav */}
        <div className="hidden md:flex items-center gap-1">
          {navLinks.map((link) => {
            const isActive =
              link.href === "/"
                ? pathname === "/"
                : pathname.startsWith(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  "px-3 py-1.5 text-sm rounded-md transition-colors",
                  isActive
                    ? "bg-primary/10 text-primary font-medium"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted"
                )}
              >
                {link.label}
              </Link>
            );
          })}
        </div>

        {/* Right side */}
        <div className="ml-auto flex items-center gap-2">
          <a
            href="https://quinnmacro.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-muted-foreground hover:text-foreground transition-colors hidden sm:inline"
          >
            quinnmacro.com
          </a>

          {/* Mobile menu */}
          <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
            <SheetTrigger
              className="md:hidden inline-flex items-center justify-center h-9 w-9 rounded-md hover:bg-muted transition-colors"
            >
              <Menu className="h-5 w-5" />
              <span className="sr-only">导航菜单</span>
            </SheetTrigger>
            <SheetContent side="left" className="w-64 p-0">
              <div className="flex flex-col py-6">
                <div className="px-6 mb-6">
                  <Link
                    href="/"
                    className="flex items-center gap-2"
                    onClick={() => setMobileOpen(false)}
                  >
                    <BarChart3 className="h-5 w-5 text-primary" />
                    <span className="font-bold text-foreground">
                      QuinnMacro
                    </span>
                  </Link>
                </div>
                <div className="flex flex-col gap-1 px-3">
                  {navLinks.map((link) => {
                    const isActive =
                      link.href === "/"
                        ? pathname === "/"
                        : pathname.startsWith(link.href);
                    return (
                      <Link
                        key={link.href}
                        href={link.href}
                        onClick={() => setMobileOpen(false)}
                        className={cn(
                          "px-3 py-2 text-sm rounded-md transition-colors",
                          isActive
                            ? "bg-primary/10 text-primary font-medium"
                            : "text-muted-foreground hover:text-foreground hover:bg-muted"
                        )}
                      >
                        <span>{link.label}</span>
                        <span className="text-xs text-muted-foreground ml-2">
                          {link.labelEn}
                        </span>
                      </Link>
                    );
                  })}
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </nav>
    </header>
  );
}
