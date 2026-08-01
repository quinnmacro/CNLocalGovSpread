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
import { VarComparisonChart } from "@/components/charts/var-comparison";
import { HillPlot } from "@/components/charts/hill-plot";
import { MeanExcessPlot } from "@/components/charts/mean-excess-plot";
import { VarBacktestChart } from "@/components/charts/var-backtest";
import { ConfidenceSlider } from "@/components/interactive/confidence-slider";
import { useRiskMetrics, useEvt, useBacktest } from "@/hooks/use-api";
import { fmt } from "@/lib/utils";
import { Loader2, AlertTriangle } from "lucide-react";

/** Inline skeleton for loading states */
function ChartSkeleton({ label }: { label: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-[350px] rounded-lg border border-dashed border-border/50 bg-muted/20">
      <Loader2 className="h-6 w-6 animate-spin text-primary/60 mb-2" />
      <span className="text-xs text-muted-foreground">{label}</span>
    </div>
  );
}

/** Inline error banner */
function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="flex items-center gap-2 p-4 rounded-lg border border-chart-3/30 bg-chart-3/5 text-sm text-muted-foreground">
      <AlertTriangle className="h-4 w-4 text-chart-3 shrink-0" />
      <span>{message}</span>
    </div>
  );
}

export function RiskContent() {
  const [confidence, setConfidence] = useState(0.99);

  const riskMetrics = useRiskMetrics(confidence);
  const evt = useEvt(0.1);
  const backtest = useBacktest(confidence);

  const hasError = riskMetrics.isError || evt.isError || backtest.isError;

  // Derived: investment implications
  const varBps = riskMetrics.data?.var_evt != null ? Math.abs(riskMetrics.data.var_evt) : null;
  const varAnnualizedBps = varBps != null ? varBps * Math.sqrt(252) : null;
  const portfolioNotional = 100_0000_0000; // 100亿面值
  const dailyLossCny = varBps != null ? (varBps / 10000) * portfolioNotional : null;

  return (
    <div className="flex min-h-[calc(100vh-3.5rem)]">
      <Sidebar />
      <div className="flex-1 px-4 md:px-8 py-8 max-w-5xl">
        <Breadcrumb
          items={[
            { label: "首页", href: "/" },
            { label: "分析", href: "/analysis/overview" },
            { label: "风险度量" },
          ]}
          className="mb-6"
        />
        <h1 className="text-3xl md:text-4xl font-bold tracking-tight mb-2">
          风险度量
        </h1>
        <p className="text-muted-foreground text-lg mb-8">
          Risk — VaR / ES / EVT 尾部分析
        </p>

        {/* Global error */}
        {hasError && (
          <ErrorBanner message="部分数据加载失败，请检查后端服务或刷新页面重试。" />
        )}

        {/* ── Section 0: WHY ── */}
        <Section
          index={0}
          title="为什么需要多种 VaR 方法？"
          subtitle="模型风险、尾部行为与监管要求"
        >
          <ProseBlock>
            <p>
              在险价值（Value at Risk, VaR）是风险管理的基石。它回答一个直觉性的问题：
              "在给定置信水平下，未来一天（或一段时期）最大可能亏损是多少？"
            </p>
            <p>
              然而，不同估计方法对收益率分布尾部的假设截然不同：
            </p>
            <ul>
              <li>
                <strong>历史模拟法（Historical Simulation）</strong>——完全依赖经验分布，
                无需参数假设，但样本外的极端事件无从体现。
              </li>
              <li>
                <strong>参数法（Parametric / Variance-Covariance）</strong>——假设正态或 t 分布，
                计算快捷，但对肥尾（fat tail）现象严重低估。
              </li>
              <li>
                <strong>极值理论（Extreme Value Theory, EVT）</strong>——
                专门建模尾部渐近行为，对极端分位数估计更稳健，
                但阈值选择和参数不确定性引入新的模型风险。
              </li>
            </ul>
            <p>
              监管框架（如巴塞尔协议 III/IV）要求银行使用 Expected Shortfall（ES）替代 VaR
              作为市场风险资本计量的主要指标，因为 ES 满足次可加性（subadditivity），
              是一致性风险度量（coherent risk measure）。比较多种方法有助于量化
              <strong>模型风险（model risk）</strong>——即因模型选择不同而产生的风险估计差异。
            </p>
          </ProseBlock>

          <InsightCard title="核心矛盾" variant="info">
            正态分布假设下，99% VaR ≈ 2.33σ；但如果收益率服从肥尾分布（tail index ξ &gt; 0），
            真实 99% VaR 可能是正态假设的 1.5-3 倍。忽略尾部行为 = 系统性低估风险。
          </InsightCard>
        </Section>

        {/* ── Section 1: HOW ── */}
        <Section
          index={1}
          title="极值理论与 GPD-POT 方法"
          subtitle="EVT 尾部建模的数学基础"
        >
          <ProseBlock>
            <p>
              极值理论（EVT）是概率论中研究随机变量极端行为的分支。
              在金融风险管理中，我们关心收益率分布的左尾（亏损尾部）。
              常用两种方法：
            </p>
            <ul>
              <li>
                <strong>区组极值法（Block Maxima, BM）</strong>——取每个区组（如每月）的最大/最小值，
                用广义极值分布（GEV）拟合。数据利用率低。
              </li>
              <li>
                <strong>超阈值法（Peaks Over Threshold, POT）</strong>——
                选取超过阈值 u 的所有观测，用广义帕累托分布（GPD）拟合超额。
                数据利用率高，更常用。
              </li>
            </ul>

            <h3>广义帕累托分布（GPD）</h3>
            <p>
              GPD 的累积分布函数（CDF）为：
            </p>
          </ProseBlock>

          <Formula
            block
            math={String.raw`G(x) = 1 - \left(1 + \xi \frac{x}{\sigma}\right)^{-1/\xi}`}
          />

          <ProseBlock>
            <p>
              其中 <Formula math={String.raw`\xi`} /> 是形状参数（shape parameter），
              <Formula math={String.raw`\sigma > 0`} /> 是尺度参数（scale parameter）。
            </p>
            <ul>
              <li>
                <Formula math={String.raw`\xi > 0`} />：Fréchet 型厚尾（金融收益率最常见）
              </li>
              <li>
                <Formula math={String.raw`\xi = 0`} />：指数型（极限情形，退化为指数分布）
              </li>
              <li>
                <Formula math={String.raw`\xi < 0`} />：Weibull 型有界尾（金融中罕见）
              </li>
            </ul>

            <h3>Hill 估计量</h3>
            <p>
              形状参数 <Formula math={String.raw`\xi`} /> 的经典估计方法是 Hill 估计量（1975）。
              设 <Formula math={String.raw`X_{(1)} \geq X_{(2)} \geq \cdots \geq X_{(n)}`} /> 为降序排列的顺序统计量，
              取前 k 个最大观测：
            </p>
          </ProseBlock>

          <Formula
            block
            math={String.raw`\hat{\xi}_{\text{Hill}} = \frac{1}{k} \sum_{i=1}^{k} \ln \frac{X_{(i)}}{X_{(k+1)}}`}
          />

          <ProseBlock>
            <p>
              Hill 估计量的关键问题是 k 的选择：k 过小导致方差大，k 过大引入偏误。
              实践中通过 Hill 图（Hill plot）寻找估计量相对稳定的区域来确定合适的 k。
            </p>

            <h3>VaR 与 ES 的 GPD 表达式</h3>
            <p>
              基于 GPD 拟合结果，VaR 和 ES 的半参数估计为：
            </p>
          </ProseBlock>

          <Formula
            block
            math={String.raw`\text{VaR}_\alpha = u + \frac{\sigma}{\xi}\left[\left(\frac{n}{N_u}(1-\alpha)\right)^{-\xi} - 1\right]`}
          />

          <Formula
            block
            math={String.raw`\text{ES}_\alpha = \frac{\text{VaR}_\alpha}{1 - \xi} + \frac{\sigma - \xi u}{1 - \xi}`}
          />

          <ProseBlock>
            <p>
              其中 u 是阈值，<Formula math={String.raw`N_u`} /> 是超过阈值的观测数，
              n 是总样本数。<Formula math={String.raw`\alpha`} /> 是置信水平。
              ES 始终大于 VaR（对于 <Formula math={String.raw`\xi < 1`} />），
              反映了尾部亏损的条件期望。
            </p>
          </ProseBlock>

          <InsightCard title="方法选择" variant="info">
            POT 方法的核心优势：只需对尾部少量超额数据拟合 GPD，不必对整个分布建模。
            但阈值 u 的选择需要经验判断——均值超额图（Mean Excess Plot）是标准工具。
          </InsightCard>
        </Section>

        {/* ── Section 2: WHAT ── */}
        <Section
          index={2}
          title="VaR 多方法对比"
          subtitle="历史模拟 vs 参数法 vs EVT"
        >
          {/* Confidence slider */}
          <div className="mb-6">
            <ConfidenceSlider
              value={confidence}
              onChange={setConfidence}
              presets={[0.95, 0.99, 0.995, 0.999]}
            />
          </div>

          {/* Loading / Error / Chart */}
          {riskMetrics.isLoading ? (
            <ChartSkeleton label="正在计算 VaR..." />
          ) : riskMetrics.isError ? (
            <ErrorBanner message="VaR 指标加载失败" />
          ) : riskMetrics.data ? (
            <>
              <VarComparisonChart
                metrics={riskMetrics.data}
                confidence={confidence}
                height={380}
              />
              <ReadGuide>
                <p>
                  柱状图展示三种方法估计的 VaR 值（取绝对值，单位 bps）。
                  通常 EVT-VaR &gt; 历史模拟 VaR &gt; 参数法 VaR（正态假设下）。
                </p>
                <p>
                  ES（Expected Shortfall）= 超过 VaR 后的条件期望亏损，
                  始终大于对应 VaR，差距越大说明尾部越厚。
                </p>
                <p>
                  拖动上方置信水平滑块观察：在更高分位数（99.5%, 99.9%）下，
                  方法间差异会急剧放大——这正是模型风险集中的区域。
                </p>
              </ReadGuide>

              {/* Metric cards */}
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-6">
                <MetricCard
                  label="历史模拟 VaR"
                  value={riskMetrics.data.var_historical}
                  unit="bps"
                  decimals={2}
                />
                <MetricCard
                  label="参数法 VaR"
                  value={riskMetrics.data.var_parametric}
                  unit="bps"
                  decimals={2}
                />
                <MetricCard
                  label="EVT VaR"
                  value={riskMetrics.data.var_evt}
                  unit="bps"
                  decimals={2}
                />
                <MetricCard
                  label="EVT ES"
                  value={riskMetrics.data.es_evt}
                  unit="bps"
                  decimals={2}
                />
                <MetricCard
                  label="VaR 方法差异"
                  value={
                    riskMetrics.data.var_evt != null && riskMetrics.data.var_parametric != null
                      ? Math.abs(riskMetrics.data.var_evt) - Math.abs(riskMetrics.data.var_parametric)
                      : null
                  }
                  unit="bps"
                  decimals={2}
                />
                <MetricCard
                  label="ES / VaR 比率"
                  value={
                    riskMetrics.data.var_evt != null && riskMetrics.data.es_evt != null && riskMetrics.data.var_evt !== 0
                      ? riskMetrics.data.es_evt / riskMetrics.data.var_evt
                      : null
                  }
                  decimals={3}
                />
              </div>

              {/* GPD parameter tooltips */}
              <div className="flex flex-wrap gap-x-6 gap-y-2 mt-6 p-4 rounded-lg bg-muted/30">
                <ParamTooltip
                  name="GPD ξ (shape)"
                  value={riskMetrics.data.gpd_shape}
                  decimals={4}
                  tooltip={
                    <>
                      广义帕累托形状参数。ξ &gt; 0 表示厚尾（Fréchet 域），
                      值越大尾部越厚。典型金融数据 ξ ∈ [0.1, 0.4]。
                      ξ &gt; 0.5 时方差不存在，ξ &gt; 1 时均值不存在。
                    </>
                  }
                />
                <ParamTooltip
                  name="GPD σ (scale)"
                  value={riskMetrics.data.gpd_scale}
                  decimals={4}
                  tooltip={
                    <>
                      广义帕累托尺度参数，控制尾部扩散速度。
                      与超额标准差相关，值越大尾部亏损分布越分散。
                    </>
                  }
                />
              </div>
            </>
          ) : null}
        </Section>

        {/* ── Section 3: SO WHAT ── */}
        <Section
          index={3}
          title="尾部诊断：Hill 图与均值超额图"
          subtitle="验证 EVT 拟合的关键参数"
        >
          <ProseBlock>
            <p>
              EVT 模型的可靠性取决于两个关键选择：
              (1) 尾指数 <Formula math={String.raw`\xi`} /> 的稳定估计，
              (2) 阈值 u 的合理选取。以下两张图是标准诊断工具。
            </p>
          </ProseBlock>

          {/* Hill Plot */}
          {evt.isLoading ? (
            <ChartSkeleton label="正在计算 Hill 估计..." />
          ) : evt.isError ? (
            <ErrorBanner message="EVT 数据加载失败" />
          ) : evt.data ? (
            <>
              <h3 className="text-lg font-semibold mt-8 mb-3">Hill 图</h3>
              <HillPlot hill={evt.data.hill} height={350} />
              <ReadGuide>
                <p>
                  横轴为 k（使用的最大观测个数），纵轴为对应的 Hill 尾指数估计
                  <Formula math={String.raw`\hat{\xi}`} />。
                </p>
                <p>
                  理想情况下应存在一段"平台区"（plateau）——在此区间内估计值相对稳定。
                  平台区对应的 <Formula math={String.raw`\xi`} /> 即为合理的尾指数。
                </p>
                <p>
                  k 过小（图左端）：估计方差大，波动剧烈。
                  k 过大（图右端）：引入非尾部数据，偏误增大。
                </p>
              </ReadGuide>

              {/* Hill summary metrics */}
              <div className="flex flex-wrap gap-x-6 gap-y-2 mt-4 p-4 rounded-lg bg-muted/30">
                <ParamTooltip
                  name="尾指数 ξ"
                  value={evt.data.hill.shape}
                  decimals={4}
                  tooltip={
                    <>
                      Hill 估计的尾指数。ξ &gt; 0 确认厚尾特征。
                      1/ξ 给出尾部幂律指数：若 1/ξ ≈ 3-5，
                      则尾部概率以多项式速率衰减（远慢于正态的指数衰减）。
                    </>
                  }
                />
                <ParamTooltip
                  name="尾部阶数 1/ξ"
                  value={evt.data.hill.shape != null && evt.data.hill.shape !== 0 ? 1 / evt.data.hill.shape : null}
                  decimals={2}
                  tooltip={
                    <>
                      尾部幂律指数 α = 1/ξ。α = 2 时方差可能不存在；
                      α = 4 时四阶矩不存在（峰度无穷）。
                      金融数据典型值 α ∈ [2, 5]。
                    </>
                  }
                />
                <ParamTooltip
                  name="阈值 (k-th order stat)"
                  value={evt.data.hill.threshold}
                  decimals={4}
                  tooltip="Hill 估计选取的阈值，即第 k+1 个最大观测值。"
                />
                <ParamTooltip
                  name="k (upper order stats)"
                  value={evt.data.hill.k}
                  decimals={0}
                  tooltip="用于 Hill 估计的上序统计量个数。在 Hill 图平台区选取。"
                />
              </div>

              {/* Mean Excess Plot */}
              <h3 className="text-lg font-semibold mt-10 mb-3">均值超额图</h3>
              <MeanExcessPlot points={evt.data.mean_excess} height={350} />
              <ReadGuide>
                <p>
                  横轴为阈值 u，纵轴为该阈值下的平均超额
                  <Formula math="E(X - u | X > u)" />。
                </p>
                <p>
                  GPD 模型成立的充要条件：当 u 足够大时，均值超额函数关于 u 是线性的。
                  因此，寻找均值超额图开始呈线性的起点作为 GPD 阈值。
                </p>
                <p>
                  红色虚线为尾部区域的线性拟合，线性程度越好说明 GPD 假设越合理。
                  非线性偏离暗示阈值选择不当或数据不满足 GPD 渐近假设。
                </p>
              </ReadGuide>

              <InsightCard title="阈值选择的影响" variant="warning">
                阈值过低 → GPD 渐近近似不成立，估计偏误大。
                阈值过高 → 超阈值样本过少，参数方差大。
                实践中常选 90%-95% 分位数作为阈值，并通过均值超额图确认线性区域。
              </InsightCard>
            </>
          ) : null}
        </Section>

        {/* ── Section 4: NOW WHAT ── */}
        <Section
          index={4}
          title="VaR 回测与投资含义"
          subtitle="模型验证与实际风控应用"
        >
          <ProseBlock>
            <p>
              模型估计完成后，必须通过回测（backtesting）验证其有效性。
              核心问题是：模型预测的覆盖率（coverage）是否与标称置信水平一致？
            </p>
            <p>
              如果 <Formula math={String.raw`\text{VaR}_{0.99}`} /> 模型正确，
              则收益率低于 −VaR 的频率应接近 1%（即约 250 个交易日中出现 2-3 次突破）。
              两个经典检验：
            </p>
            <ul>
              <li>
                <strong>Kupiec 检验（1995）</strong>——似然比检验，
                验证违规频率是否等于预期频率（如 1%）。
                原假设：实际覆盖率 = 标称覆盖率。
              </li>
              <li>
                <strong>Christoffersen 检验（1998）</strong>——
                不仅检验覆盖率，还检验违规事件的独立性（无聚集效应）。
                违规聚集说明模型未能捕捉波动率聚集。
              </li>
            </ul>
          </ProseBlock>

          {/* Backtest chart */}
          {backtest.isLoading ? (
            <ChartSkeleton label="正在运行回测..." />
          ) : backtest.isError ? (
            <ErrorBanner message="回测数据加载失败" />
          ) : backtest.data ? (
            <>
              <VarBacktestChart
                varSeries={backtest.data.var_series}
                violations={backtest.data.violations}
                nObservations={backtest.data.n_observations}
                actualCoverage={backtest.data.actual_coverage}
                kupiec={backtest.data.kupiec}
                christoffersen={backtest.data.christoffersen}
                height={400}
              />
              <ReadGuide>
                <p>
                  图中曲线为滚动 VaR 估计值（取绝对值），
                  柱状标记为实际亏损超过 VaR 的"违规"事件。
                </p>
                <p>
                  违规次数应与预期接近：{fmt(confidence * 100, 1)}% 置信水平下，
                  {backtest.data.n_observations} 个交易日预期约{" "}
                  {fmt(backtest.data.expected_violations, 1)} 次违规。
                </p>
                <p>
                  Kupiec p-value &gt; 0.05 → 不能拒绝覆盖率正确假设。
                  Christoffersen p-value &gt; 0.05 → 不能拒绝独立性假设。
                  两者同时通过 → 模型可信。
                </p>
              </ReadGuide>

              {/* Backtest stats */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
                <MetricCard
                  label="违规次数"
                  value={backtest.data.violations}
                  unit={`次 / ${backtest.data.n_observations}天`}
                  decimals={0}
                />
                <MetricCard
                  label="实际覆盖率"
                  value={backtest.data.actual_coverage * 100}
                  unit="%"
                  decimals={2}
                />
                <MetricCard
                  label="Kupiec p-value"
                  value={backtest.data.kupiec.pvalue}
                  decimals={4}
                />
                <MetricCard
                  label="Christoffersen p-value"
                  value={backtest.data.christoffersen.pvalue}
                  decimals={4}
                />
              </div>

              {/* Pass/fail badges */}
              <div className="flex flex-wrap gap-3 mt-4">
                <span
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border ${
                    backtest.data.passes
                      ? "border-chart-2/30 bg-chart-2/10 text-chart-2"
                      : "border-chart-3/30 bg-chart-3/10 text-chart-3"
                  }`}
                >
                  {backtest.data.passes ? "✓" : "✗"} 回测综合判定：
                  {backtest.data.passes ? "通过" : "未通过"}
                </span>
                <span
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border ${
                    backtest.data.kupiec.pvalue > 0.05
                      ? "border-chart-2/30 bg-chart-2/10 text-chart-2"
                      : "border-chart-3/30 bg-chart-3/10 text-chart-3"
                  }`}
                >
                  {backtest.data.kupiec.pvalue > 0.05 ? "✓" : "✗"} Kupiec 覆盖率检验
                </span>
                <span
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border ${
                    backtest.data.christoffersen.pvalue > 0.05
                      ? "border-chart-2/30 bg-chart-2/10 text-chart-2"
                      : "border-chart-3/30 bg-chart-3/10 text-chart-3"
                  }`}
                >
                  {backtest.data.christoffersen.pvalue > 0.05 ? "✓" : "✗"} Christoffersen 独立性检验
                </span>
              </div>

              {/* Investment implications */}
              <InsightCard title="投资含义" variant="success">
                <div className="space-y-3">
                  <p>
                    以 EVT-VaR 为基准风险度量：
                    {varBps != null ? (
                      <>
                        <strong className="font-mono text-foreground">
                          {" "}{fmt(varBps, 2)} bps/天
                        </strong>
                        ，即每个交易日有 {fmt((1 - confidence) * 100, 1)}% 概率
                        亏损超过 {fmt(varBps, 2)} bps。
                      </>
                    ) : (
                      "数据加载中..."
                    )}
                  </p>
                  {varAnnualizedBps != null && (
                    <p>
                      年化风险（假设独立同分布，√252 缩放）：
                      <strong className="font-mono text-foreground">
                        {" "}约 {fmt(varAnnualizedBps, 0)} bps
                      </strong>
                      （即 {fmt(varAnnualizedBps / 100, 2)}%）。
                    </p>
                  )}
                  {dailyLossCny != null && (
                    <p>
                      持仓 100 亿面值的单日 VaR 约为：
                      <strong className="font-mono text-foreground">
                        {" "}¥{fmt(dailyLossCny / 10000, 0)} 万
                      </strong>
                      。这意味着在极端情景下（{(1 - confidence) * 100}% 概率），
                      一天内可能亏损超过此金额。
                    </p>
                  )}
                  <p className="text-xs text-muted-foreground/80 mt-2">
                    注意：VaR 只给出"亏损不超过 X"的概率保证，
                    不描述超过 VaR 后的亏损分布。ES 补充了这一信息——
                    超过 VaR 后的条件期望亏损通常比 VaR 大 20-50%。
                  </p>
                </div>
              </InsightCard>
            </>
          ) : null}
        </Section>

        {/* ── Footer spacer ── */}
        <div className="h-16" />
      </div>
    </div>
  );
}
