"use client";

import { useQuery } from "@tanstack/react-query";
import { Sidebar } from "@/components/layout/sidebar";
import { Breadcrumb } from "@/components/layout/breadcrumb";
import { Section } from "@/components/narrative/section";
import { ProseBlock } from "@/components/narrative/prose-block";
import { ReadGuide } from "@/components/narrative/read-guide";
import { InsightCard } from "@/components/narrative/insight-card";
import { MetricCard } from "@/components/narrative/metric-card";
import { ParamTooltip } from "@/components/narrative/param-tooltip";
import { TimeSeriesChart } from "@/components/charts/time-series";
import { DistributionChart } from "@/components/charts/distribution";
import { TermStructureChart } from "@/components/charts/term-structure";
import { DataTable } from "@/components/interactive/data-table";
import { Skeleton } from "@/components/ui/skeleton";
import { useDataStatistics, useDataSummary } from "@/hooks/use-api";
import { api } from "@/lib/api";
import { fmt } from "@/lib/utils";
import type { TimePoint } from "@/lib/types";
import Link from "next/link";
import { ArrowRight } from "lucide-react";

const PAGE_INFO = {
  title: "利差全景",
  subtitle: "Overview — 趋势、分布、期限结构、分布特征",
};

export function OverviewContent() {
  const { data: stats, isLoading: statsLoading, error: statsError } = useDataStatistics();
  const { data: summary, isLoading: summaryLoading, error: summaryError } = useDataSummary();
  const { data: rawData, isLoading: rawLoading, error: rawError } = useQuery({
    queryKey: ["data", "raw", 2500, 0],
    queryFn: () => api.dataRaw(2500, 0),
    staleTime: 10 * 60 * 1000,
  });

  const isLoading = statsLoading || summaryLoading || rawLoading;
  const hasError = statsError || summaryError || rawError;

  // Transform raw data into time series for each column
  const timeSeriesData = rawData?.data.map((row) => ({
    date: String(row.date),
    spread_all: Number(row.spread_all),
    spread_5y: Number(row.spread_5y),
    spread_10y: Number(row.spread_10y),
    spread_30y: Number(row.spread_30y),
  }));

  const series5y: TimePoint[] = timeSeriesData?.map((d) => ({ date: d.date, value: d.spread_5y })) ?? [];
  const series10y: TimePoint[] = timeSeriesData?.map((d) => ({ date: d.date, value: d.spread_10y })) ?? [];
  const series30y: TimePoint[] = timeSeriesData?.map((d) => ({ date: d.date, value: d.spread_30y })) ?? [];
  const seriesAll: TimePoint[] = timeSeriesData?.map((d) => ({ date: d.date, value: d.spread_all })) ?? [];

  // Extract values for distribution chart
  const values5y = series5y.map((p) => p.value);
  const values10y = series10y.map((p) => p.value);
  const values30y = series30y.map((p) => p.value);

  // Key events for time series annotation
  const events = [
    { date: "2020-01-23", label: "武汉封城" },
    { date: "2020-03-09", label: "美股熔断" },
    { date: "2022-11-11", label: "防疫二十条" },
  ];

  if (isLoading) {
    return (
      <div className="flex min-h-[calc(100vh-3.5rem)]">
        <Sidebar />
        <div className="flex-1 px-4 md:px-8 py-8 max-w-5xl">
          <Skeleton className="h-8 w-64 mb-2" />
          <Skeleton className="h-6 w-96 mb-8" />
          <div className="space-y-8">
            <Skeleton className="h-64 w-full" />
            <Skeleton className="h-96 w-full" />
            <Skeleton className="h-96 w-full" />
          </div>
        </div>
      </div>
    );
  }

  if (hasError) {
    return (
      <div className="flex min-h-[calc(100vh-3.5rem)]">
        <Sidebar />
        <div className="flex-1 px-4 md:px-8 py-8 max-w-5xl">
          <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-8 text-center">
            <p className="text-destructive font-medium">数据加载失败</p>
            <p className="text-sm text-muted-foreground mt-2">请检查后端 API 连接或稍后重试</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-[calc(100vh-3.5rem)]">
      <Sidebar />
      <div className="flex-1 px-4 md:px-8 py-8 max-w-5xl">
        <Breadcrumb
          items={[
            { label: "首页", href: "/" },
            { label: "分析", href: "/analysis/overview" },
            { label: PAGE_INFO.title },
          ]}
          className="mb-6"
        />
        <h1 className="text-3xl md:text-4xl font-bold tracking-tight mb-2">
          {PAGE_INFO.title}
        </h1>
        <p className="text-muted-foreground text-lg mb-8">{PAGE_INFO.subtitle}</p>

        {/* Section 0: WHY — 研究动机 */}
        <Section index={0} title="为何分析城投债信用利差？">
          <ProseBlock>
            <p>
              中国城投债市场规模超过 <strong>40 万亿元</strong>，是地方政府隐性债务的核心载体。
              信用利差（Credit Spread）定义为城投债收益率与同期限国债收益率之差，反映了市场对
              信用风险的定价。利差的动态变化不仅影响债券定价，更是系统性风险的重要先行指标。
            </p>
            <p>
              本模块从 <strong>趋势、分布、期限结构、统计特征</strong> 四个维度全面刻画利差行为，
              为后续波动率建模、风险度量和情景分析奠定基础。
            </p>
          </ProseBlock>
        </Section>

        {/* Section 1: HOW — 方法论 */}
        <Section index={1} title="数据来源与统计概览" subtitle="Data & Summary Statistics">
          <ProseBlock>
            <p>
              数据来自 Wind EDB，时间跨度 {summary?.date_range[0]?.slice(0, 10)} 至 {summary?.date_range[1]?.slice(0, 10)}，
              共 {summary?.n_rows} 个交易日，包含 4 个利差序列：全品种、5Y、10Y、30Y。
            </p>
          </ProseBlock>

          {stats && <DataTable columns={stats.columns} />}

          <ReadGuide>
            <p>
              <strong>表格字段说明：</strong>N 为样本量；均值/标准差单位为 bps；偏度 &gt; 0 表示右偏，
              峰度 &gt; 3 表示尖峰（Leptokurtic）；ADF p &lt; 0.05 拒绝单位根假设，判定为平稳序列。
            </p>
          </ReadGuide>

          {stats && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
              {stats.columns.map((col) => (
                <MetricCard
                  key={col.column}
                  label={col.column}
                  value={col.mean}
                  unit="bps"
                  decimals={2}
                />
              ))}
            </div>
          )}
        </Section>

        {/* Section 2: WHAT — 结果展示 */}
        <Section index={2} title="利差可视化" subtitle="Trends, Distribution & Term Structure">
          <h3 className="text-lg font-semibold text-foreground mb-3 mt-8">
            时序走势与关键事件
          </h3>
          <TimeSeriesChart
            series={[
              { name: "全品种", data: seriesAll },
              { name: "5Y", data: series5y },
              { name: "10Y", data: series10y },
              { name: "30Y", data: series30y },
            ]}
            events={events}
            height={500}
          />
          <ReadGuide>
            <p>
              <strong>读图要点：</strong>四条曲线展示不同期限利差走势。虚线标注重大事件节点，
              观察利差在压力时期的跳升行为（Flight-to-Quality）。2020 年初疫情冲击导致利差
              短期走阔，随后在宽松政策下快速收窄。
            </p>
          </ReadGuide>

          <h3 className="text-lg font-semibold text-foreground mb-3 mt-12">
            利差分布：KDE + 直方图
          </h3>
          <DistributionChart
            series={[
              { name: "5Y", values: values5y },
              { name: "10Y", values: values10y },
              { name: "30Y", values: values30y },
            ]}
            height={450}
          />
          <ReadGuide>
            <p>
              <strong>读图要点：</strong>直方图为经验分布，实线为核密度估计（KDE），虚线为正态参考。
              注意尾部厚度（Tail Thickness）和尖峰特征（Leptokurtosis），这些偏离正态的特征
              对后续 GARCH 建模和 EVT 尾部分析至关重要。
            </p>
          </ReadGuide>

          <h3 className="text-lg font-semibold text-foreground mb-3 mt-12">
            期限结构：5Y vs 10Y / 30Y 散点图
          </h3>
          <TermStructureChart
            series5y={series5y}
            series10y={series10y}
            series30y={series30y}
            height={450}
          />
          <ReadGuide>
            <p>
              <strong>读图要点：</strong>散点图展示期限利差的协动关系。45° 虚线为等利差参考线。
              若点云集中在 45° 线上方，说明长端利差更高（正常期限结构）；若分散或下穿，
              则可能存在期限溢价异常或流动性溢价压缩。
            </p>
          </ReadGuide>
        </Section>

        {/* Section 3: SO WHAT — 诊断与解读 */}
        <Section index={3} title="分布特征与建模含义">
          {stats && stats.columns.length > 0 && (
            <InsightCard title="尖峰厚尾（Leptokurtic）" variant="warning">
              <p>
                所有序列的超额峰度（Excess Kurtosis）均 &gt; 3，其中
                {stats.columns.map((col, i) => (
                  <span key={col.column}>
                    {" "}
                    <ParamTooltip
                      name={col.column}
                      value={col.kurtosis}
                      decimals={2}
                      tooltip={
                        <>
                          峰度 = {fmt(col.kurtosis, 2)}，远超正态分布基准 3。
                          表明利差分布存在显著厚尾，极端事件概率高于正态假设。
                        </>
                      }
                    />
                    {i < stats.columns.length - 1 && "、"}
                  </span>
                ))}
                。这意味着：
              </p>
              <ul className="list-disc list-inside mt-2 space-y-1">
                <li>
                  <strong>GARCH 族模型</strong>可捕捉波动率聚集（Vol Clustering），但正态残差假设不足；
                </li>
                <li>
                  需引入 <strong>t 分布 / 偏 t 分布</strong> 拟合厚尾；
                </li>
                <li>
                  极端风险度量需依赖 <strong>EVT（极值理论）</strong> 而非参数 VaR。
                </li>
              </ul>
            </InsightCard>
          )}

          {stats && stats.columns.length > 0 && (
            <InsightCard title="平稳性检验（ADF Test）" variant="info">
              <p>
                增强 Dickey-Fuller 检验结果显示：
                {stats.columns.map((col) => (
                  <span key={col.column}>
                    {" "}
                    <ParamTooltip
                      name={col.column}
                      value={col.adf_pvalue}
                      decimals={4}
                      tooltip={
                        <>
                          ADF p-value = {fmt(col.adf_pvalue, 4)}。
                          {col.is_stationary
                            ? "p < 0.05，拒绝单位根假设，序列平稳。"
                            : "p ≥ 0.05，无法拒绝单位根假设，序列可能非平稳。"}
                        </>
                      }
                    />
                    、
                  </span>
                ))}
                平稳性是时间序列建模的前提，非平稳序列需差分或协整处理。
              </p>
            </InsightCard>
          )}
        </Section>

        {/* Section 4: NOW WHAT — 投资含义 */}
        <Section index={4} title="下一步：波动率建模">
          <ProseBlock>
            <p>
              厚尾特征和波动率聚集现象表明，简单的历史标准差无法准确刻画利差风险。
              下一步将使用 GARCH 族模型（GARCH / EGARCH / GJR-GARCH / FIGARCH）拟合条件波动率，
              并通过 AIC / BIC 信息准则和残差诊断选择最优模型。
            </p>
          </ProseBlock>
          <Link
            href="/analysis/volatility"
            className="inline-flex items-center gap-2 mt-6 px-4 py-2 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            进入波动率建模
            <ArrowRight className="h-4 w-4" />
          </Link>
        </Section>
      </div>
    </div>
  );
}
