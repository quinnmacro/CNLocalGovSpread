import { Separator } from "@/components/ui/separator";

export function Footer() {
  return (
    <footer className="border-t border-border/40 mt-auto">
      <div className="max-w-7xl mx-auto px-4 md:px-6 py-6">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-muted-foreground">
          <div className="flex items-center gap-2">
            <span>© {new Date().getFullYear()}</span>
            <Separator orientation="vertical" className="h-3" />
            <a
              href="https://quinnmacro.com"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-foreground transition-colors"
            >
              QuinnMacro
            </a>
          </div>
          <div className="flex items-center gap-4">
            <span>中国地方政府债信用利差量化分析平台</span>
            <Separator orientation="vertical" className="h-3" />
            <span>数据来源: Wind EDB</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
