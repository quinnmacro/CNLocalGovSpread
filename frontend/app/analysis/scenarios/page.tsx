import { Breadcrumb } from "@/components/layout/breadcrumb";
import { Sidebar } from "@/components/layout/sidebar";

const titles: Record<string, { title: string; subtitle: string }> = {
  overview: { title: "利差全景", subtitle: "Overview — 趋势、分布、期限结构、分布特征" },
  volatility: { title: "波动率建模", subtitle: "Volatility — GARCH 族模型对比与诊断" },
  risk: { title: "风险度量", subtitle: "Risk — VaR / ES / EVT 尾部分析" },
  regimes: { title: "市场状态", subtitle: "Regimes — HMM + MarketGauge 状态识别" },
  scenarios: { title: "情景分析", subtitle: "Scenarios — 蒙特卡洛模拟与压力测试" },
};

const pageInfo = titles["scenarios"];

export const metadata = { title: pageInfo.title };

export default function Page() {
  return (
    <div className="flex min-h-[calc(100vh-3.5rem)]">
      <Sidebar />
      <div className="flex-1 px-4 md:px-8 py-8 max-w-5xl">
        <Breadcrumb
          items={[
            { label: "首页", href: "/" },
            { label: "分析", href: "/analysis/overview" },
            { label: pageInfo.title },
          ]}
          className="mb-6"
        />
        <h1 className="text-3xl md:text-4xl font-bold tracking-tight mb-2">
          {pageInfo.title}
        </h1>
        <p className="text-muted-foreground text-lg mb-8">{pageInfo.subtitle}</p>
        <div className="rounded-lg border border-dashed border-border/50 p-12 text-center text-muted-foreground">
          <p className="text-sm font-mono">Phase 3–4: 即将实现完整叙事与图表</p>
        </div>
      </div>
    </div>
  );
}
