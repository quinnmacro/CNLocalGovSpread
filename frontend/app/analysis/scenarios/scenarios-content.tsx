"use client";

import { useState } from "react";
import { Sidebar } from "@/components/layout/sidebar";
import { Breadcrumb } from "@/components/layout/breadcrumb";
import { Section } from "@/components/narrative/section";
import { ProseBlock } from "@/components/narrative/prose-block";
import { Formula } from "@/components/narrative/formula";
import { ReadGuide } from "@/components/narrative/read-guide";
import { InsightCard } from "@/components/narrative/insight-card";
import { MetricCard } from "@/components/narrative/metric-card";
import { ParamTooltip } from "@/components/narrative/param-tooltip";
import { FanChart } from "@/components/charts/fan-chart";
import { StressTable } from "@/components/charts/stress-table";
import { HorizonSlider } from "@/components/interactive/horizon-slider";
import { Skeleton } from "@/components/ui/skeleton";
import { useScenarios, useStress } from "@/hooks/use-api";
import { fmt } from "@/lib/utils";
import Link from "next/link";
import { ArrowRight } from "lucide-react";

const PAGE_INFO = {
  title: "情景分析",
  subtitle: "Scenarios — 蒙特卡洛模拟与压力测试",
};

export function ScenariosContent() {
  const [horizon, setHorizon] = useState(252);
  const { data: scenarios, isLoading: scenariosLoading, error: scenariosError } = useScenarios(horizon, 5000);
  const stress = useStress();

  const isLoading = scenariosLoading;
  const hasError = scenariosError;

  // Trigger stress test when scenarios load
  const runStressTest = () => {
    if (scenarios) {
      stress.mutate({
        current: scenarios.current_spread,
        shock_multipliers: [1.0, 1.5, 2.0, 3.0],
        horizon: horizon,
        n_paths: 5000,
      });
    }
  };

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
        <Section index={0} title="为何需要情景分析？">
          <ProseBlock>
            <p>
              历史回测（Backtesting）告诉我们模型在过去表现如何，但投资组合管理是<strong>前瞻性</strong>的。
              基金经理需要回答：<em>"未来 N 天，利差可能走阔到多少？极端情景下的损失有多大？"</em>
            </p>
            <p>
              <strong>蒙特卡洛模拟</strong>（Monte Carlo Simulation）通过生成大量随机路径，
              构建利差的前瞻分布（Forward Distribution）。与单点预测不同，它给出完整的
              概率分布，包括中位数、5% 分位数（乐观情景）、95% 分位数（悲观情景）。
            </p>
            <p>
              <strong>压力测试</strong>（Stress Testing）则假设波动率放大（如 1.5x、2x、3x），
              模拟极端市场条件下的利差走势。这回答了：<em>"如果发生类似 2020 年新冠或 2022 年理财赎回潮，
              组合可能面临多大亏损？"</em>
            </p>
          </ProseBlock>
        </Section>

        {/* Section 1: HOW — 方法论 */}
        <Section index={1} title="AR(1) + GARCH 模拟框架">
          <ProseBlock>
            <p>
              蒙特卡洛模拟基于前面拟合的 <strong>AR(1) + GARCH</strong> 模型。利差的一阶差分
              <Formula math="\Delta y_t" /> 服从 AR(1) 过程，条件方差由 GARCH 模型给出：
            </p>
          </ProseBlock>

          <div className="my-6 p-4 rounded-lg bg-muted/30">
            <Formula
              block
              math={String.raw`\begin{aligned}
\Delta y_t &= \phi \cdot \Delta y_{t-1} + \varepsilon_t \\
\varepsilon_t &= \sigma_t \cdot z_t, \quad z_t \sim t(\nu) \\
\sigma_t^2 &= \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2
\end{aligned}`}
            />
          </div>

          <ProseBlock>
            <p>
              模拟步骤：
            </p>
            <ol className="list-decimal list-inside space-y-1">
              <li>
                从 GARCH 模型提取参数 <Formula math="\omega, \alpha, \beta, \nu" /> 和最近的
                条件方差 <Formula math="\sigma_T^2" />；
              </li>
              <li>
                生成 <Formula math="N" /> 条路径，每条路径 <Formula math="H" /> 天（horizon）；
              </li>
              <li>
                每天从 <Formula math="t(\nu)" /> 分布抽样标准化残差 <Formula math="z_t" />，
                计算 <Formula math="\varepsilon_t" /> 和 <Formula math="\sigma_t^2" />；
              </li>
              <li>
                累加得到 <Formula math="y_{T+H}" /> 的分布，提取分位数。
              </li>
            </ol>
          </ProseBlock>

          <ProseBlock>
            <p>
              <strong>压力测试</strong>通过将波动率参数乘以倍数（如 1.5x、2x、3x）实现。
              这等价于假设未来波动率高于历史平均水平，模拟极端情景。
            </p>
          </ProseBlock>
        </Section>

        {/* Section 2: WHAT — 结果展示 */}
        <Section index={2} title="蒙特卡洛模拟结果">
          {scenarios && (
            <>
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-foreground">
                  扇形图（Fan Chart）：{horizon} 天前瞻
                </h3>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">模拟路径：</span>
                  <span className="font-mono text-sm">{scenarios.n_paths.toLocaleString()}</span>
                </div>
              </div>

              <HorizonSlider
                value={horizon}
                onChange={setHorizon}
                min={10}
                max={500}
                step={1}
                presets={[
                  { value: 30, label: "1月" },
                  { value: 60, label: "3月" },
                  { value: 120, label: "6月" },
                  { value: 252, label: "1年" },
                ]}
              />

              <div className="mt-6">
                <FanChart
                  currentSpread={scenarios.current_spread}
                  horizon={scenarios.horizon}
                  median={scenarios.median_final}
                  p5={scenarios.p5_final}
                  p95={scenarios.p95_final}
                  height={450}
                />
              </div>

              <ReadGuide>
                <p>
                  <strong>读图要点：</strong>扇形图显示利差的前瞻分布。中心线为中位数预测，
                  上下边界分别为 95% 和 5% 分位数。
                </p>
                <p>
                  扇形宽度随时间递增，反映预测不确定性增加。窄扇形（如 30 天）表示短期预测
                  相对可靠，宽扇形（如 252 天）表示长期预测不确定性大。
                </p>
                <p>
                  关注 95% 分位数（上边界）：这是"悲观情景"下的利差水平。若该值显著高于当前利差，
                  说明存在较大的走阔风险。
                </p>
              </ReadGuide>

              {/* Quantile metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
                <MetricCard
                  label="当前利差"
                  value={scenarios.current_spread}
                  unit="bps"
                  decimals={2}
                />
                <MetricCard
                  label="中位数预测"
                  value={scenarios.median_final}
                  unit="bps"
                  decimals={2}
                />
                <MetricCard
                  label="5% 分位数"
                  value={scenarios.p5_final}
                  unit="bps"
                  decimals={2}
                />
                <MetricCard
                  label="95% 分位数"
                  value={scenarios.p95_final}
                  unit="bps"
                  decimals={2}
                />
              </div>

              <InsightCard title="分位数解读" variant="info">
                <p>
                  在未来 <strong>{horizon}</strong> 天内：
                </p>
                <ul className="list-disc list-inside mt-2 space-y-1">
                  <li>
                    <strong>50% 概率</strong>：利差将低于 <span className="font-mono">{fmt(scenarios.median_final, 2)} bps</span>
                    （中位数预测）；
                  </li>
                  <li>
                    <strong>90% 概率</strong>：利差将落在 <span className="font-mono">{fmt(scenarios.p5_final, 2)}</span> ~
                    <span className="font-mono"> {fmt(scenarios.p95_final, 2)} bps</span> 区间内；
                  </li>
                  <li>
                    <strong>5% 概率</strong>（极端情景）：利差将超过 <span className="font-mono">{fmt(scenarios.p95_final, 2)} bps</span>。
                  </li>
                </ul>
              </InsightCard>
            </>
          )}
        </Section>

        {/* Section 3: SO WHAT — 诊断与解读 */}
        <Section index={3} title="压力测试：波动率放大情景">
          {scenarios && (
            <>
              <ProseBlock>
                <p>
                  压力测试假设波动率高于历史平均水平。通过将 GARCH 参数 <Formula math="\omega" /> 和
                  <Formula math="\alpha" /> 乘以倍数，模拟极端市场条件下的利差走势。
                </p>
              </ProseBlock>

              <div className="mt-6">
                <button
                  onClick={runStressTest}
                  disabled={stress.isPending}
                  className="px-4 py-2 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {stress.isPending ? "计算中..." : "运行压力测试"}
                </button>
              </div>

              {stress.isSuccess && stress.data && (
                <div className="mt-8">
                  <h3 className="text-lg font-semibold text-foreground mb-4">
                    压力情景对比
                  </h3>
                  <StressTable scenarios={stress.data.scenarios} />

                  <ReadGuide>
                    <p>
                      <strong>读图要点：</strong>表格显示不同波动率倍数下的利差预测。
                      "波动率倍数" 列表示假设波动率是历史平均的多少倍（如 2x 表示波动率翻倍）。
                    </p>
                    <p>
                      关注 "95% 分位数" 列：这是极端情景下的利差上限。"超越概率" 列表示
                      利差超过当前水平的概率。概率越高，说明走阔风险越大。
                    </p>
                  </ReadGuide>

                  <InsightCard title="压力情景的投资含义" variant="warning">
                    <p>
                      假设当前持仓 <strong>100 亿面值</strong> 城投债，久期约 <strong>5 年</strong>。
                      利差走阔 <strong>100 bps</strong> 时，组合亏损约：
                    </p>
                    <p className="mt-2 p-3 rounded bg-muted/50 font-mono text-sm">
                      亏损 ≈ 100 亿 × 5 年 × 100 bps = <strong>50 亿</strong>
                    </p>
                    <p className="mt-2">
                      因此，压力测试的 95% 分位数直接对应组合的<strong>在险价值</strong>（VaR）。
                      基金经理需确保有足够的资本缓冲覆盖此类极端亏损。
                    </p>
                  </InsightCard>
                </div>
              )}

              {stress.isError && (
                <div className="mt-6 rounded-lg border border-destructive/50 bg-destructive/10 p-4">
                  <p className="text-destructive text-sm">压力测试计算失败，请检查后端 API</p>
                </div>
              )}
            </>
          )}
        </Section>

        {/* Section 4: NOW WHAT — 投资含义 */}
        <Section index={4} title="综合投资建议">
          <ProseBlock>
            <p>
              蒙特卡洛模拟和压力测试为投资组合管理提供了前瞻性的量化工具：
            </p>
            <ul className="list-disc list-inside space-y-2 mt-3">
              <li>
                <strong>扇形图</strong>给出利差的概率分布，帮助设定止损和止盈水平；
              </li>
              <li>
                <strong>压力测试</strong>评估极端情景下的组合亏损，指导资本配置；
              </li>
              <li>
                <strong>分位数</strong>（5%、50%、95%）为情景分析提供框架，支持决策树分析。
              </li>
            </ul>
          </ProseBlock>

          <InsightCard title="行动建议" variant="success">
            <ul className="list-disc list-inside space-y-2">
              <li>
                若 95% 分位数显著高于当前利差（如 &gt; 50 bps），建议降低久期敞口或增加对冲；
              </li>
              <li>
                压力测试的 2x 波动率情景可作为<strong>资本充足率</strong>的参考基准；
              </li>
              <li>
                定期（如每周）更新模拟结果，动态调整组合权重；
              </li>
              <li>
                结合 HMM 状态识别：高波动期使用更保守的压力倍数（如 3x），低波动期可用 1.5x。
              </li>
            </ul>
          </InsightCard>

          <ProseBlock>
            <p>
              至此，我们完成了从数据探索、波动率建模、风险度量、状态识别到情景分析的完整研究流程。
              下一步可将这些模型集成到自动化交易系统中，实现动态风险管理和组合优化。
            </p>
          </ProseBlock>
          <Link
            href="/analysis/overview"
            className="inline-flex items-center gap-2 mt-6 px-4 py-2 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            返回利差全景
            <ArrowRight className="h-4 w-4" />
          </Link>
        </Section>
      </div>
    </div>
  );
}
