"use client";
import { useState, useEffect, useMemo } from "react";

import { Sidebar } from "@/components/layout/sidebar";
import { Breadcrumb } from "@/components/layout/breadcrumb";
import { Section } from "@/components/narrative/section";
import { ProseBlock } from "@/components/narrative/prose-block";
import { Formula } from "@/components/narrative/formula";
import { ReadGuide } from "@/components/narrative/read-guide";
import { InsightCard } from "@/components/narrative/insight-card";
import { MetricCard } from "@/components/narrative/metric-card";
import { ParamTooltip } from "@/components/narrative/param-tooltip";
import { ChartWrapper } from "@/components/narrative/chart-wrapper";
import { RegimeSequenceChart } from "@/components/charts/regime-sequence";
import { TransitionHeatmap } from "@/components/charts/transition-heatmap";
import { MarketGaugePanel } from "@/components/charts/market-gauge-panel";
import { STSSignalChart } from "@/components/charts/sts-signal";
import { BayesianSTSChart } from "@/components/charts/bayesian-sts";
import { ChangepointChart } from "@/components/charts/changepoint";
import { KalmanSignalChart } from "@/components/charts/kalman-signal";
import { Skeleton } from "@/components/ui/skeleton";
import { ExecutiveSummary } from "@/components/narrative/executive-summary";
import { PageNavigation } from "@/components/narrative/page-navigation";
import {
  useHmm,
  useMarketGauge,
  useKalmanSignal,
  useStsSignal,
  useBayesianSts,
  useChangepoints,
} from "@/hooks/use-api";
import type { TimePoint } from "@/lib/types";
import { useScrollSpy } from "@/hooks/use-scroll-spy";
import { fmt } from "@/lib/utils";
import Link from "next/link";
import {
  ArrowRight,
  Activity,
  Brain,
  TrendingUp,
  Layers,
  Shield,
  ChevronUp,
  Zap,
  Target,
  BarChart3,
} from "lucide-react";

const PAGE_INFO = {
  title: "市场状态",
  subtitle: "Regimes — HMM + Kalman + STS + Bayesian + CPD 多维状态识别",
};

const TOC_SECTIONS = [
  { id: "why", label: "WHY", sublabel: "研究动机" },
  { id: "how", label: "HOW", sublabel: "方法论" },
  { id: "what", label: "WHAT", sublabel: "结果展示" },
  { id: "so-what", label: "SO WHAT", sublabel: "诊断" },
  { id: "now-what", label: "NOW WHAT", sublabel: "投资" },
] as const;

const SECTION_IDS = TOC_SECTIONS.map((s) => s.id) as unknown as readonly string[];

/** Reconstruct raw spread from signal + deviation */
function reconstructSpread(
  signal: TimePoint[],
  deviation: TimePoint[],
): TimePoint[] {
  const sigMap = new Map(signal.map((s) => [s.date, s.value]));
  return deviation.map((d) => ({
    date: d.date,
    value: d.value + (sigMap.get(d.date) ?? 0),
  }));
}

function getSignalVerdict(
  stsZ: number,
  bayesZ: number,
  isOverSts: boolean,
  isOverBayes: boolean,
): { label: string; className: string; icon: typeof TrendingUp } {
  if (isOverSts && isOverBayes)
    return {
      label: "利差高估",
      className: "verdict-badge verdict-overvalued",
      icon: TrendingUp,
    };
  if (
    (stsZ < -1.5 && bayesZ < -1.5) ||
    (!isOverSts && !isOverBayes && stsZ < -1.5)
  )
    return {
      label: "利差低估",
      className: "verdict-badge verdict-undervalued",
      icon: Activity,
    };
  return {
    label: "利差合理",
    className: "verdict-badge verdict-neutral",
    icon: Shield,
  };
}

export function RegimesContent() {
  const { data: hmm, isLoading: hmmLoading, error: hmmError } = useHmm(3);
  const {
    data: gauge,
    isLoading: gaugeLoading,
    error: gaugeError,
  } = useMarketGauge();
  const { data: sts, isLoading: stsLoading, error: stsError } = useStsSignal();
  const {
    data: bayes,
    isLoading: bayesLoading,
    error: bayesError,
  } = useBayesianSts();
  const {
    data: cpd,
    isLoading: cpdLoading,
    error: cpdError,
  } = useChangepoints("spread_all", "binseg", 5);
  const {
    data: kalman,
    isLoading: kalmanLoading,
  } = useKalmanSignal();

  const isLoading = hmmLoading || gaugeLoading || kalmanLoading || stsLoading || bayesLoading;
  const activeSection = useScrollSpy(SECTION_IDS, 120, !isLoading);

  const [showBackToTop, setShowBackToTop] = useState(false);
  useEffect(() => {
    const onScroll = () => setShowBackToTop(window.scrollY > 400);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  const hasError = hmmError || gaugeError;

  const events = [
    { date: "2020-01-23", label: "武汉封城" },
    { date: "2020-03-09", label: "美股熔断" },
    { date: "2022-11-11", label: "防疫二十条" },
    { date: "2022-11-14", label: "理财赎回潮" },
  ];

  // Compute derived values for signal dashboard
  const signalDashboard = useMemo(() => {
    if (!sts || !bayes || !hmm) return null;
    const lastStsZ = sts.deviation_zscore[sts.deviation_zscore.length - 1]?.value ?? 0;
    const lastBayesZ = bayes.deviation_zscore[bayes.deviation_zscore.length - 1]?.value ?? 0;
    const lastKalmanZ = kalman?.deviation_zscore[kalman.deviation_zscore.length - 1]?.value ?? 0;
    const currentRegime = hmm.labels[hmm.labels.length - 1]?.regime ?? 0;
    const verdict = getSignalVerdict(
      lastStsZ,
      lastBayesZ,
      sts.is_overvalued,
      bayes.is_overvalued,
    );
    return { lastStsZ, lastBayesZ, lastKalmanZ, currentRegime, verdict };
  }, [sts, bayes, hmm, kalman]);

  if (isLoading) {
    return (
      <div className="flex min-h-[calc(100vh-3.5rem)]">
        <Sidebar />
        <div className="flex-1 px-4 md:px-8 py-8 max-w-5xl">
          <Skeleton className="h-8 w-64 mb-2" />
          <Skeleton className="h-6 w-96 mb-8" />
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
            <Skeleton className="h-24 w-full rounded-xl" />
            <Skeleton className="h-24 w-full rounded-xl" />
            <Skeleton className="h-24 w-full rounded-xl" />
            <Skeleton className="h-24 w-full rounded-xl" />
          </div>
          <Skeleton className="h-32 w-full rounded-xl mb-8" />
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
          <h1 className="text-2xl font-bold mb-4">{PAGE_INFO.title}</h1>
          <p className="text-destructive">
            数据加载失败。请检查后端 API 是否运行。
          </p>
        </div>
      </div>
    );
  }

  // Derived values
  const lastStsZ =
    sts?.deviation_zscore[sts.deviation_zscore.length - 1]?.value ?? 0;
  const lastBayesZ =
    bayes?.deviation_zscore[bayes.deviation_zscore.length - 1]?.value ?? 0;
  const lastKalmanZ =
    kalman?.deviation_zscore[kalman.deviation_zscore.length - 1]?.value ?? 0;
  const currentRegime = hmm?.labels[hmm.labels.length - 1]?.regime ?? 0;
  const regimeNames = ["低波动 (Calm)", "中波动 (Caution)", "高波动 (Stress)"];
  const spread =
    sts && bayes
      ? reconstructSpread(sts.signal, sts.deviation)
      : [];

  const verdict = getSignalVerdict(
    lastStsZ,
    lastBayesZ,
    sts?.is_overvalued ?? false,
    bayes?.is_overvalued ?? false,
  );
  const VerdictIcon = verdict.icon;

  // Bayesian CI width for metric strip
  const bayesCIWidth = bayes
    ? (bayes.signal_upper[bayes.signal_upper.length - 1]?.value ?? 0) -
      (bayes.signal_lower[bayes.signal_lower.length - 1]?.value ?? 0)
    : 0;

  // HMM regime stats for sigma display
  const regimeStats = hmm?.regime_stats ?? [];
  const currentSigma = regimeStats.find((s) => s.regime === currentRegime)?.std ?? 0;

  return (
    <div className="flex min-h-[calc(100vh-3.5rem)]">
      <Sidebar />
      <div className="flex-1 px-4 md:px-8 py-8 max-w-5xl">
        <Breadcrumb items={[{ label: "分析", href: "/analysis" }, { label: "市场状态" }]} />

        {/* ── Hero Header ─────────────────────────────────────────── */}
        <header className="mb-10 relative overflow-hidden rounded-2xl">
          {/* Animated gradient background */}
          <div className="absolute inset-0 hero-gradient-bg pointer-events-none" />
          <div className="relative z-10 px-6 py-8 md:px-8 md:py-10">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h1 className="text-3xl md:text-4xl font-bold hero-gradient tracking-tight">
                  {PAGE_INFO.title}
                </h1>
                <p className="text-muted-foreground mt-2 text-sm md:text-base max-w-xl leading-relaxed">
                  {PAGE_INFO.subtitle}
                </p>
              </div>
              {signalDashboard && (
                <div className="hidden md:flex items-center gap-2">
                  <VerdictIcon className={`h-4 w-4 ${
                    signalDashboard.verdict.label === "利差高估"
                      ? "text-chart-3"
                      : signalDashboard.verdict.label === "利差低估"
                      ? "text-chart-2"
                      : "text-muted-foreground"
                  }`} />
                  <span className={signalDashboard.verdict.className}>
                    {signalDashboard.verdict.label}
                  </span>
                </div>
              )}
            </div>

            {/* ── Metric Strip ────────────────────────────────────────── */}
            {hmm && sts && gauge && bayes && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-6">
                <MetricCard
                  label="当前状态 (HMM)"
                  value={regimeNames[currentRegime]}
                  accent="pink"
                  sparkline={hmm.transition_matrix[currentRegime] ?? []}
                />
                <MetricCard
                  label="偏离度 (STS z-score)"
                  value={`${lastStsZ >= 0 ? "+" : ""}${fmt(lastStsZ, 3)}`}
                  accent="cyan"
                  sparkline={sts.deviation_zscore.slice(-30).map((d) => d.value)}
                />
                <MetricCard
                  label="MarketGauge"
                  value={fmt(gauge.composite, 1)}
                  accent="orange"
                  sparkline={Object.values(gauge.indicators).map((ind) => ind.score)}
                />
                <MetricCard
                  label="Bayesian CI 宽度"
                  value={`${fmt(bayesCIWidth, 1)} bps`}
                  accent="blue"
                  sparkline={bayes.signal_upper
                    .slice(-30)
                    .map((u, i) => u.value - (bayes.signal_lower.slice(-30)[i]?.value ?? 0))}
                />
              </div>
            )}
          </div>
        </header>

        {/* ── Signal Dashboard ─────────────────────────────────────── */}
        {sts && bayes && hmm && (
          <div className="signal-dashboard mb-10">
            <div className="signal-dashboard-header">
              <div className="flex items-center gap-2.5">
                <Zap className="h-4 w-4 text-primary" />
                <h2 className="text-sm font-semibold text-foreground tracking-tight">
                  综合信号仪表盘
                </h2>
              </div>
              <span className="text-[10px] text-muted-foreground font-mono uppercase tracking-widest">
                Signal Aggregation
              </span>
            </div>
            <div className="signal-dashboard-grid">
              <div className="signal-card">
                <div className="signal-card-label">STS 信号</div>
                <div className={`signal-card-value ${
                  sts.is_overvalued ? "text-chart-3" : sts.is_undervalued ? "text-chart-2" : "text-foreground"
                }`}>
                  {sts.is_overvalued ? "高估" : sts.is_undervalued ? "低估" : "合理"}
                </div>
                <div className="signal-card-detail">
                  z = {lastStsZ >= 0 ? "+" : ""}{fmt(lastStsZ, 3)}
                </div>
              </div>
              {kalman && (
                <div className="signal-card">
                  <div className="signal-card-label">Kalman 滤波</div>
                  <div className={`signal-card-value ${
                    kalman.is_overvalued ? "text-chart-3" : kalman.is_undervalued ? "text-chart-2" : "text-foreground"
                  }`}>
                    {kalman.is_overvalued ? "高估" : kalman.is_undervalued ? "低估" : "合理"}
                  </div>
                  <div className="signal-card-detail">
                    z = {lastKalmanZ >= 0 ? "+" : ""}{fmt(lastKalmanZ, 3)}
                  </div>
                </div>
              )}
              <div className="signal-card">
                <div className="signal-card-label">Bayesian STS</div>
                <div className={`signal-card-value ${
                  bayes.is_overvalued ? "text-chart-3" : bayes.is_undervalued ? "text-chart-2" : "text-foreground"
                }`}>
                  {bayes.is_overvalued ? "高估" : bayes.is_undervalued ? "低估" : "合理"}
                </div>
                <div className="signal-card-detail">
                  z = {lastBayesZ >= 0 ? "+" : ""}{fmt(lastBayesZ, 3)}
                </div>
              </div>
              <div className="signal-card">
                <div className="signal-card-label">HMM 状态</div>
                <div className={`signal-card-value ${
                  currentRegime === 2 ? "text-chart-3" : currentRegime === 1 ? "text-chart-3" : "text-chart-2"
                }`}>
                  {regimeNames[currentRegime]}
                </div>
                <div className="signal-card-detail">
                  σ = {fmt(currentSigma, 4)}
                </div>
              </div>
              <div className={`signal-card signal-card-verdict ${
                verdict.label === "利差高估" ? "signal-card-overvalued" :
                verdict.label === "利差低估" ? "signal-card-undervalued" : "signal-card-neutral"
              }`}>
                <div className="signal-card-label">综合判断</div>
                <div className="signal-card-value flex items-center gap-1.5">
                  <VerdictIcon className="h-3.5 w-3.5" />
                  {verdict.label}
                </div>
                <div className="signal-card-detail">
                  {verdict.label === "利差高估" ? "多方法一致：利差偏窄" :
                   verdict.label === "利差低估" ? "多方法一致：利差偏宽" : "利差处于合理范围"}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── Sticky TOC ───────────────────────────────────────────── */}
        <nav className="toc-nav">
          {TOC_SECTIONS.map((s) => (
            <a
              key={s.id}
              href={`#${s.id}`}
              className={activeSection === s.id ? "active" : ""}
            >
              <span className="toc-label">{s.label}</span>
              <span className="toc-sublabel">{s.sublabel}</span>
            </a>
          ))}
        </nav>

        {/* ── Section 1: WHY ───────────────────────────────────────── */}
        <Section id="why" index={0} title="为何需要状态识别？">
          <ProseBlock>
            <p>
              中国城投债利差在 2020-2026 年间经历了多次剧烈波动——从疫情冲击、
              理财赎回潮到政策转向。传统的线性模型无法捕捉这些{" "}
              <strong>结构性变化</strong>（structural breaks）和{" "}
              <strong>状态切换</strong>（regime switches）。
            </p>
            <p>
              状态识别的核心问题是：利差当前处于什么 &quot;世界&quot;？
              是低波动的平稳期，还是高波动的压力期？利差是否偏离了基本面趋势？
              这些判断直接影响配置策略和对冲决策。
            </p>
          </ProseBlock>
          <ProseBlock variant="callout">
            <p>
              本页使用五种互补方法进行多维状态分析：
            </p>
            <ul className="mt-2 space-y-1.5">
              <li>
                <strong>HMM</strong>（隐马尔可夫模型）— 将波动率聚类为隐含状态
              </li>
              <li>
                <strong>Kalman Filter</strong>（卡尔曼滤波）— Local Level Model 提取平滑趋势
              </li>
              <li>
                <strong>STS</strong>（结构化时间序列）— 提取趋势信号并量化偏离度
              </li>
              <li>
                <strong>Bayesian STS</strong>（贝叶斯结构化时间序列）— MCMC 推断后验不确定性
              </li>
              <li>
                <strong>Change Point Detection</strong>（结构性变化点）— 定位均值断裂的时间点
              </li>
            </ul>
          </ProseBlock>
        </Section>

        <div className="section-divider" />

        {/* ── Section 2: HOW ───────────────────────────────────────── */}
        <Section id="how" index={1} title="方法论：五种状态识别模型">
          {/* Kalman Filter */}
          <h3 className="text-lg font-semibold text-foreground mb-3">
            0. 卡尔曼滤波信号提取（Kalman Filter）
          </h3>
          <ProseBlock>
            <p>
              经典 Local Level Model 将利差分解为不可观测的趋势信号{" "}
              <Formula math="\mu_t" /> 和短期噪声 <Formula math="\varepsilon_t" />：
            </p>
          </ProseBlock>
          <Formula
            block
            math="\begin{aligned}
y_t &= \mu_t + \varepsilon_t \quad &\text{（观测方程）}\\
\mu_{t+1} &= \mu_t + \eta_t \quad &\text{（状态方程）}
\end{aligned}"
          />
          <ProseBlock>
            <p>
              信噪比{" "}
              <ParamTooltip
                name="Q"
                value={kalman ? `${kalman.q_ratio > 1e6 ? ">>1" : kalman.q_ratio.toFixed(2)}` : null}
                tooltip={
                  <p>
                    Q = σ²_η / σ²_ε 决定滤波平滑度。Q → 0 时信号几乎不变（强均值回归），
                    Q → ∞ 时信号 ≈ 原始数据（无滤波）。适中的 Q 值（0.01-1）提供最佳信号提取。
                  </p>
                }
              />{" "}
              = <Formula math="\sigma^2_\eta / \sigma^2_\varepsilon" />{" "}
              决定了滤波的平滑度。Kalman 递推通过预测-更新两步迭代，
              实时估计不可观测的趋势状态。
            </p>
          </ProseBlock>
          <ReadGuide>
            <p>
              <strong>与 STS 的关系</strong>：Local Level Model 是 STS 的简化版——
              没有 slope（漂移率）成分，因此更适合短期平稳信号提取，
              而 STS 更适合有趋势的长期序列。
            </p>
          </ReadGuide>

          {/* HMM */}
          <h3 className="text-lg font-semibold text-foreground mt-8 mb-3">
            1. 隐马尔可夫模型（Hidden Markov Model）
          </h3>
          <ProseBlock>
            <p>
              HMM 假设利差的波动率由{" "}
              <Formula math="K" /> 个隐含状态之一生成，状态之间按转移概率矩阵{" "}
              <Formula math="P" /> 切换：
            </p>
          </ProseBlock>
          <Formula
            block
            math="P = \begin{pmatrix} p_{11} & p_{12} & \cdots \\ p_{21} & p_{22} & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix}, \quad \sum_j p_{ij} = 1"
          />
          <ReadGuide>
            <p>
              对角线元素 <Formula math="p_{ii}" />{" "}
              表示状态持续性。值越高，状态越稳定。
              非对角线元素表示状态切换概率。例如 <Formula math="p_{12}" />{" "}
              是从低波动切换到中波动的概率。
            </p>
          </ReadGuide>

          {/* STS */}
          <h3 className="text-lg font-semibold text-foreground mt-8 mb-3">
            2. 结构化时间序列（Structural Time Series）
          </h3>
          <ProseBlock>
            <p>STS 将利差分解为不可观测的成分：</p>
          </ProseBlock>
          <Formula
            block
            math="\begin{aligned}
y_t &= \mu_t + \varepsilon_t \quad &\text{（观测方程）}\\
\mu_{t+1} &= \mu_t + \nu_t + \eta_t \quad &\text{（level + drift）}\\
\nu_{t+1} &= \nu_t + \zeta_t \quad &\text{（slope 随机游走）}
\end{aligned}"
          />
          <ProseBlock>
            <p>
              相比经典 Kalman Filter（Local Level Model），STS 额外引入了{" "}
              <ParamTooltip
                name="drift/slope"
                value={sts?.slope[sts.slope.length - 1]?.value ?? null}
                tooltip={
                  <p>
                    slope（漂移率）衡量趋势变化的速度。正值表示利差趋势上升（收窄），
                    负值表示趋势下降（走阔）。slope 的变化可以提前预警趋势反转。
                  </p>
                }
              />{" "}
              成分，可以捕捉趋势的加速和减速。通过 Kalman 滤波 + 最大似然估计，
              得到平滑的趋势信号和标准化偏离度（z-score）。
            </p>
          </ProseBlock>

          {/* Bayesian STS */}
          <h3 className="text-lg font-semibold text-foreground mt-8 mb-3">
            3. 贝叶斯结构化时间序列（Bayesian STS）
          </h3>
          <ProseBlock>
            <p>
              在经典 STS 基础上，使用 <strong>PyMC</strong> 进行贝叶斯推断，
              通过 MCMC（NUTS 采样器）获得所有参数的后验分布：
            </p>
          </ProseBlock>
          <Formula
            block
            math="\begin{aligned}
\sigma_\eta &\sim \text{HalfNormal}(1) \\
\sigma_\varepsilon &\sim \text{HalfNormal}(1) \\
\mu_t &\sim \mathcal{N}(\mu_{t-1}, \sigma_\eta^2) \\
y_t &\sim \mathcal{N}(\mu_t, \sigma_\varepsilon^2)
\end{aligned}"
          />
          <ProseBlock>
            <p>
              贝叶斯方法的核心优势是提供{" "}
              <strong>后验不确定性量化</strong>——每个时点的信号估计都伴随一个
              完整的概率分布，而非单一点估计。这使得我们可以计算{" "}
              <ParamTooltip
                name="80% CI"
                value={bayes ? `${fmt(bayesCIWidth, 1)} bps` : null}
                tooltip={
                  <p>
                    80% 后验置信区间（HDI）表示趋势估计的不确定性范围。
                    区间宽度扩大通常出现在市场结构变化期（如政策冲击、流动性危机），
                    反映了模型对新数据的不确定性上升。
                  </p>
                }
              />{" "}
              宽度，为风险评估提供概率化的决策依据。
            </p>
          </ProseBlock>

          {/* Change Point Detection */}
          <h3 className="text-lg font-semibold text-foreground mt-8 mb-3">
            4. 结构性变化点检测（Change Point Detection）
          </h3>
          <ProseBlock>
            <p>
              变化点检测从不同角度审视利差：不是识别 &quot;状态&quot;，
              而是定位均值发生 <strong>结构性断裂</strong> 的时间点。
              使用 Binary Segmentation 算法递归地最小化全局代价函数：
            </p>
          </ProseBlock>
          <Formula
            block
            math="\mathcal{C}(y) = \min_{\tau_1, \ldots, \tau_K} \sum_{k=0}^{K} \text{cost}(y_{\tau_k:\tau_{k+1}}) + \beta \cdot K"
          />
          <ReadGuide>
            <p>
              <strong>与 HMM 的互补关系</strong>：HMM 识别 &quot;状态&quot;（当前市场是什么 regime），
              而 CPD 识别 &quot;转折点&quot;（regime 何时发生变化）。
              两者结合可以同时回答 &quot;现在是什么&quot; 和 &quot;什么时候变的&quot;。
            </p>
          </ReadGuide>
        </Section>

        <div className="section-divider" />

        {/* ── Section 3: WHAT ──────────────────────────────────────── */}
        <Section id="what" index={2} title="结果：多维状态可视化">
          {/* HMM Regime Sequence */}
          {/* ── Executive Summary ─────────────────────────────────── */}
          {sts && bayes && kalman && gauge && hmm && (
            <ExecutiveSummary
              kalmanZ={lastKalmanZ}
              stsZ={lastStsZ}
              bayesZ={lastBayesZ}
              currentRegime={regimeNames[currentRegime]}
              gaugeComposite={gauge.composite}
              kalmanOvervalued={kalman.is_overvalued}
              stsOvervalued={sts.is_overvalued}
              bayesOvervalued={bayes.is_overvalued}
            />
          )}

          {hmm && (
            <>
              <h3 className="text-lg font-semibold text-foreground mb-3">
                HMM 状态时序与转移矩阵
              </h3>
              <ChartWrapper
                title="HMM 状态序列"
                subtitle="背景色 = 隐含状态，折线 = 利差"
                icon={<Brain className="h-4 w-4" />}
                metrics={[
                  { label: "当前状态", value: regimeNames[currentRegime], color: "var(--chart-4)" },
                  { label: "σ", value: fmt(currentSigma, 4) },
                ]}
              >
                <RegimeSequenceChart
                  labels={hmm.labels}
                  spread={spread.length > 0 ? spread : undefined}
                  nRegimes={hmm.n_regimes}
                  events={events}
                />
              </ChartWrapper>
              <ReadGuide>
                <p>
                  <strong>背景色带</strong>表示 HMM 推断的隐含状态。
                  绿色 = 低波动（Calm），蓝色 = 中波动（Caution），
                  红色 = 高波动（Stress）。叠加的折线为原始利差。
                </p>
                <p>
                  关注高波动期的持续时间和切换频率。
                  如果 <Formula math="p_{33}" />{" "}
                  很高（热力图右下角），说明高波动状态具有很强的持续性，
                  一旦进入就不容易退出。
                </p>
              </ReadGuide>

              <ChartWrapper
                title="转移概率矩阵"
                subtitle="行 = 当前状态，列 = 下一期状态"
                icon={<Layers className="h-4 w-4" />}
                className="mt-6"
              >
                <TransitionHeatmap matrix={hmm.transition_matrix} />
              </ChartWrapper>
              <ReadGuide>
                <p>
                  <strong>对角线</strong>值越高，状态越持久。
                  关注 <Formula math="p_{13}" />（低→高）和{" "}
                  <Formula math="p_{31}" />（高→低）——
                  这两个值决定了市场在高/低波动之间切换的速度。
                </p>
              </ReadGuide>
            </>
          )}

          {/* Kalman Signal */}
          {kalman && spread.length > 0 && (
            <>
              <h3 className="text-lg font-semibold text-foreground mt-10 mb-3">
                Kalman 滤波信号提取
              </h3>
              <ChartWrapper
                title="Kalman Filter (Local Level Model)"
                subtitle="上: 趋势信号 | 下: z-score + ±1.5 阈值"
                icon={<Target className="h-4 w-4" />}
                metrics={[
                  {
                    label: "z-score",
                    value: `${lastKalmanZ >= 0 ? "+" : ""}${fmt(lastKalmanZ, 3)}`,
                    color: Math.abs(lastKalmanZ) > 1.5 ? "var(--chart-3)" : "var(--chart-2)",
                  },
                  {
                    label: "Q (信噪比)",
                    value: kalman.q_ratio > 1e6 ? ">>1" : fmt(kalman.q_ratio, 2),
                  },
                ]}
              >
                <KalmanSignalChart
                  spread={spread}
                  signal={kalman.signal}
                  deviationZscore={kalman.deviation_zscore}
                  events={events}
                />
              </ChartWrapper>
              <ReadGuide>
                <p>
                  <strong>上面板</strong>：蓝色细线为原始利差，绿色粗线为 Kalman
                  滤波提取的平滑趋势（Local Level 信号）。
                </p>
                <p>
                  <strong>下面板</strong>：z-score 量化利差偏离趋势的程度。
                  红色背景 = 高估区间（z {">"} 1.5），绿色背景 = 低估区间（z {"<"} -1.5）。
                </p>
                <p>
                  与 STS 对比：Kalman 使用 Local Level Model（无 drift），
                  STS 使用 Local Linear Trend（有 drift），因此 Kalman 信号更平滑
                  但对趋势变化的响应更慢。
                </p>
              </ReadGuide>
            </>
          )}

          {/* STS Signal */}
          {sts && (
            <>
              <h3 className="text-lg font-semibold text-foreground mt-10 mb-3">
                STS 信号提取与偏离度分析
              </h3>
              <ChartWrapper
                title="Structural Time Series 信号"
                subtitle="上: 趋势信号 | 中: 漂移率 | 下: z-score"
                icon={<Target className="h-4 w-4" />}
                metrics={[
                  {
                    label: "z-score",
                    value: `${lastStsZ >= 0 ? "+" : ""}${fmt(lastStsZ, 3)}`,
                    color: Math.abs(lastStsZ) > 1.5 ? "var(--chart-3)" : "var(--chart-2)",
                  },
                  {
                    label: "slope",
                    value: fmt(sts.slope[sts.slope.length - 1]?.value ?? 0, 4),
                  },
                ]}
              >
                <STSSignalChart
                  spread={spread}
                  signal={sts.signal}
                  slope={sts.slope}
                  deviationZscore={sts.deviation_zscore}
                  events={events}
                />
              </ChartWrapper>
              <ReadGuide>
                <p>
                  <strong>上面板</strong>：蓝色折线为原始利差，
                  绿色粗线为 STS 提取的趋势信号（level 成分）。
                </p>
                <p>
                  <strong>中间板</strong>：漂移率（slope）衡量趋势变化的速度。
                  正值 = 利差趋势收窄，负值 = 趋势走阔。
                </p>
                <p>
                  <strong>下面板</strong>：z-score 量化利差偏离趋势的程度。
                  超过 ±1.5 阈值（虚线）表示显著偏离。
                  红色背景 = 高估区间，绿色背景 = 低估区间。
                </p>
              </ReadGuide>
            </>
          )}

          {/* Bayesian STS */}
          {bayes && (
            <>
              <h3 className="text-lg font-semibold text-foreground mt-10 mb-3">
                Bayesian STS 后验推断
              </h3>
              <ChartWrapper
                title="Bayesian STS 信号与置信区间"
                subtitle="上: 趋势 + 80% CI | 下: z-score"
                icon={<BarChart3 className="h-4 w-4" />}
                metrics={[
                  {
                    label: "z-score",
                    value: `${lastBayesZ >= 0 ? "+" : ""}${fmt(lastBayesZ, 3)}`,
                    color: Math.abs(lastBayesZ) > 1.5 ? "var(--chart-4)" : "var(--chart-2)",
                  },
                  {
                    label: "CI 宽度",
                    value: `${fmt(bayesCIWidth, 1)} bps`,
                  },
                ]}
              >
                <BayesianSTSChart
                  spread={spread}
                  signal={bayes.signal}
                  signalLower={bayes.signal_lower}
                  signalUpper={bayes.signal_upper}
                  deviationZscore={bayes.deviation_zscore}
                  events={events}
                />
              </ChartWrapper>
              <ReadGuide>
                <p>
                  <strong>上面板</strong>：绿色填充区域为 80% 后验置信区间（HDI）。
                  区间越宽，表示趋势估计的不确定性越大。
                </p>
                <p>
                  <strong>下面板</strong>：与经典 STS 的 z-score 对比，
                  Bayesian 版本的 z-score 考虑了参数不确定性，
                  通常更保守（绝对值更小）。
                </p>
              </ReadGuide>
            </>
          )}

          {/* Change Point Detection */}
          {cpd && spread.length > 0 && (
            <>
              <h3 className="text-lg font-semibold text-foreground mt-10 mb-3">
                结构性变化点检测
              </h3>
              <ChartWrapper
                title="Binary Segmentation 变化点"
                subtitle="垂直虚线 = 均值断裂点，彩色段 = 分段均值 ± σ"
                icon={<Activity className="h-4 w-4" />}
                metrics={[
                  { label: "断裂点", value: `${cpd.breakpoint_dates.length} 个` },
                  { label: "算法", value: "binseg" },
                ]}
              >
                <ChangepointChart
                  spread={spread}
                  breakpointDates={cpd.breakpoint_dates}
                  segments={cpd.segments}
                  events={events}
                />
              </ChartWrapper>
              <ReadGuide>
                <p>
                  <strong>垂直虚线</strong>标注了 Binary Segmentation 算法检测到的均值断裂点。
                  每个彩色水平线代表一个分段内的均值水平（μ），
                  浅色背景区域表示 ±1σ 范围。
                </p>
                <p>
                  与 HMM 对比：HMM 可能将同一 &quot;状态&quot; 内的不同均值水平
                  归为同一类，而 CPD 能更精确地定位均值跳跃的时刻。
                </p>
              </ReadGuide>
            </>
          )}
        </Section>

        <div className="section-divider" />

        {/* ── Section 3.5: SO WHAT ─────────────────────────────── */}
        <Section id="so-what" index={3} title="诊断与解读：多方法交叉验证">
          {sts && bayes && hmm && (
            <>
              <h3 className="text-lg font-semibold text-foreground mb-3">
                方法对比：不同视角看同一问题
              </h3>
              <div className="method-comparison-table">
                <table className="comparison-table">
                  <thead>
                    <tr>
                      <th>方法</th>
                      <th>核心输出</th>
                      <th>当前信号</th>
                      <th>优势</th>
                      <th>局限</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>HMM</td>
                      <td>隐含状态 + 转移概率</td>
                      <td>
                        <span className={currentRegime === 2 ? "text-chart-3" : currentRegime === 1 ? "text-chart-3" : "text-chart-2"}>
                          {regimeNames[currentRegime]}
                        </span>
                      </td>
                      <td>捕捉波动率聚类</td>
                      <td>预设状态数量</td>
                    </tr>
                    <tr>
                      <td>Kalman</td>
                      <td>平滑趋势 + z-score</td>
                      <td>
                        <span className={Math.abs(lastKalmanZ) > 1.5 ? (lastKalmanZ > 0 ? "text-chart-3" : "text-chart-2") : "text-muted-foreground"}>
                          z = {lastKalmanZ >= 0 ? "+" : ""}{fmt(lastKalmanZ, 2)}
                        </span>
                      </td>
                      <td>简单高效</td>
                      <td>无趋势成分</td>
                    </tr>
                    <tr>
                      <td>STS</td>
                      <td>趋势 + z-score</td>
                      <td>
                        <span className={Math.abs(lastStsZ) > 1.5 ? (lastStsZ > 0 ? "text-chart-3" : "text-chart-2") : "text-muted-foreground"}>
                          z = {lastStsZ >= 0 ? "+" : ""}{fmt(lastStsZ, 2)}
                        </span>
                      </td>
                      <td>量化偏离度</td>
                      <td>线性假设</td>
                    </tr>
                    <tr>
                      <td>Bayesian STS</td>
                      <td>后验分布 + CI</td>
                      <td>
                        <span className={Math.abs(lastBayesZ) > 1.5 ? (lastBayesZ > 0 ? "text-chart-3" : "text-chart-2") : "text-muted-foreground"}>
                          z = {lastBayesZ >= 0 ? "+" : ""}{fmt(lastBayesZ, 2)}
                        </span>
                      </td>
                      <td>不确定性量化</td>
                      <td>计算成本高</td>
                    </tr>
                    <tr>
                      <td>CPD</td>
                      <td>断裂时间点</td>
                      <td>
                        <span className="text-muted-foreground">
                          {cpd?.breakpoint_dates.length ?? 0} 个断裂点
                        </span>
                      </td>
                      <td>精确定位转折</td>
                      <td>不预测方向</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <InsightCard
                title="交叉验证结论"
                variant={
                  verdict.label === "利差高估"
                    ? "warning"
                    : verdict.label === "利差低估"
                    ? "success"
                    : "info"
                }
              >
                <p>
                  五种方法从不同角度描述了当前市场状态：
                </p>
                <ul className="list-disc list-inside mt-2 space-y-1.5">
                  <li>
                    <strong>HMM</strong> 将市场归类为{" "}
                    <span className="font-medium">{regimeNames[currentRegime]}</span>
                    {currentRegime === 2 && "，表明市场处于压力期"}
                    {currentRegime === 0 && "，表明市场平稳运行"}
                  </li>
                  <li>
                    <strong>Kalman</strong> 滤波的 z-score（{fmt(lastKalmanZ, 2)}）
                    {Math.abs(lastKalmanZ) > 1.5
                      ? lastKalmanZ > 0
                        ? "显示利差显著高于趋势（高估）"
                        : "显示利差显著低于趋势（低估）"
                      : "显示利差处于合理范围"}
                  </li>
                  <li>
                    <strong>STS</strong> 的 z-score（{fmt(lastStsZ, 2)}）显示利差
                    {Math.abs(lastStsZ) > 1.5
                      ? lastStsZ > 0
                        ? "显著高于趋势（高估）"
                        : "显著低于趋势（低估）"
                      : "处于合理范围"}
                  </li>
                  <li>
                    <strong>Bayesian STS</strong> 的 80% CI 宽度为{" "}
                    {fmt(bayesCIWidth, 1)} bps，
                    {bayesCIWidth > 20
                      ? "不确定性较高，需关注"
                      : "估计精度较好"}
                  </li>
                  {cpd && cpd.breakpoint_dates.length > 0 && (
                    <li>
                      <strong>CPD</strong> 检测到最近一次均值断裂发生在{" "}
                      <span className="font-mono">
                        {cpd.breakpoint_dates[cpd.breakpoint_dates.length - 1]}
                      </span>
                      {(() => {
                        const segs = cpd.segments;
                        if (segs.length < 2) return null;
                        const last = segs[segs.length - 1];
                        const prev = segs[segs.length - 2];
                        return (
                          <>
                            ，之后均值从 {fmt(prev.mean, 1)} 变为{" "}
                            {fmt(last.mean, 1)} bps（
                            {last.mean > prev.mean ? "走阔" : "收窄"}{" "}
                            {fmt(Math.abs(last.mean - prev.mean), 1)} bps）
                          </>
                        );
                      })()}
                    </li>
                  )}
                </ul>
              </InsightCard>
            </>
          )}
        </Section>

        <div className="section-divider" />

        {/* ── Section 4: NOW WHAT ────────────────────────────── */}
        <Section
          id="now-what"
          index={4}
          title="MarketGauge 仪表盘与投资建议"
        >
          {gauge && (
            <>
              <h3 className="text-lg font-semibold text-foreground mb-3">
                综合市场状态仪表盘
              </h3>
              <MarketGaugePanel gauge={gauge} />
              <ReadGuide>
                <p>
                  <strong>读图要点：</strong>MarketGauge 将多维市场指标聚合为
                  单一分数（0-100）。分数越高，市场压力越大。状态标签
                  （Calm/Caution/Stress/Crisis）基于分数阈值划分。
                </p>
                <p>
                  子指标的进度条显示各维度的相对位置。
                  综合判断时需关注：(1) 分数趋势（上升/下降），(2)
                  各子指标是否同步恶化。
                </p>
              </ReadGuide>

              <InsightCard
                title={`当前状态：${gauge.status[1]}（${gauge.status[0]}）`}
                variant={
                  gauge.composite < 30
                    ? "success"
                    : gauge.composite < 60
                    ? "warning"
                    : "info"
                }
              >
                <p>
                  MarketGauge 综合分数为{" "}
                  <strong className="font-mono">
                    {fmt(gauge.composite, 1)}
                  </strong>
                  ，对应状态为 <strong>{gauge.status[1]}</strong>。
                </p>
                <ul className="list-disc list-inside mt-3 space-y-1">
                  {gauge.composite < 30 && (
                    <>
                      <li>市场处于低压力状态，可适度增加风险敞口</li>
                      <li>利差收窄趋势持续，城投债配置价值较高</li>
                      <li>建议关注期限利差套利机会</li>
                    </>
                  )}
                  {gauge.composite >= 30 && gauge.composite < 60 && (
                    <>
                      <li>市场进入警戒区间，需密切监控波动率和流动性</li>
                      <li>建议降低久期敞口，增加短端配置</li>
                      <li>对冲成本上升，需评估 CDS 保护策略</li>
                    </>
                  )}
                  {gauge.composite >= 60 && (
                    <>
                      <li>市场压力显著，建议防御性配置</li>
                      <li>利差可能快速走阔，避免追涨杀跌</li>
                      <li>关注政策干预信号，等待稳定后再加仓</li>
                    </>
                  )}
                </ul>
              </InsightCard>

              <InsightCard
                title="HMM + Kalman + MarketGauge + STS + Bayesian 联合信号"
                variant="info"
              >
                <p>
                  当 HMM 识别出高波动状态且 MarketGauge 分数 &gt; 60 且 STS
                  z-score &gt; 1.5 且 Bayesian CI 扩大时，四重信号确认市场处于
                  压力期。此时应：(1) 降低仓位至 50% 以下，(2)
                  增加高评级城投债占比，(3) 监控转移概率{" "}
                  <Formula math="p_{31}" />
                  （高→低波动）以判断拐点，(4) 等待 STS z-score 回归至 ±1.0
                  以内且 Bayesian CI 收窄后再考虑加仓。
                </p>
              </InsightCard>
            </>
          )}

          <ProseBlock>
            <p>
              状态识别和综合仪表盘为动态资产配置提供了量化依据。下一步将使用
              蒙特卡洛模拟生成利差的前瞻分布，评估不同情景下的组合风险。
            </p>
          </ProseBlock>
          <PageNavigation
            prev={{ href: "/analysis/risk", label: "风险度量", emoji: "⚠️" }}
            next={{ href: "/analysis/scenarios", label: "情景分析", emoji: "🔮" }}
          />
        </Section>
      </div>
      {/* Back to top */}
      {showBackToTop && (
        <button
          className="back-to-top"
          onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
          aria-label="返回顶部"
        >
          <ChevronUp className="h-5 w-5" />
        </button>
      )}
    </div>
  );
}
