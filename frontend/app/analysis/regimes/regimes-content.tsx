"use client";

import { Sidebar } from "@/components/layout/sidebar";
import { Breadcrumb } from "@/components/layout/breadcrumb";
import { Section } from "@/components/narrative/section";
import { ProseBlock } from "@/components/narrative/prose-block";
import { Formula } from "@/components/narrative/formula";
import { ReadGuide } from "@/components/narrative/read-guide";
import { InsightCard } from "@/components/narrative/insight-card";
import { MetricCard } from "@/components/narrative/metric-card";
import { ParamTooltip } from "@/components/narrative/param-tooltip";
import { RegimeSequenceChart } from "@/components/charts/regime-sequence";
import { TransitionHeatmap } from "@/components/charts/transition-heatmap";
import { MarketGaugePanel } from "@/components/charts/market-gauge-panel";
import { Skeleton } from "@/components/ui/skeleton";
import { useHmm, useMarketGauge } from "@/hooks/use-api";
import { fmt } from "@/lib/utils";
import Link from "next/link";
import { ArrowRight } from "lucide-react";

const PAGE_INFO = {
  title: "市场状态",
  subtitle: "Regimes — HMM + MarketGauge 状态识别",
};

export function RegimesContent() {
  const { data: hmm, isLoading: hmmLoading, error: hmmError } = useHmm(3);
  const { data: gauge, isLoading: gaugeLoading, error: gaugeError } = useMarketGauge();

  const isLoading = hmmLoading || gaugeLoading;
  const hasError = hmmError || gaugeError;

  // Key events for regime chart annotation
  const events = [
    { date: "2020-01-23", label: "武汉封城" },
    { date: "2020-03-09", label: "美股熔断" },
    { date: "2022-11-11", label: "防疫二十条" },
    { date: "2022-11-14", label: "理财赎回潮" },
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
        <Section index={0} title="为何需要状态识别？">
          <ProseBlock>
            <p>
              金融时间序列往往不是平稳的（Non-stationary），而是存在<strong>结构性断裂</strong>（Structural Breaks）
              和<strong>状态切换</strong>（Regime Switching）。例如，市场在"低波动平稳期"和"高波动危机期"之间
              交替，不同状态下的收益率分布、波动率水平、相关性结构都截然不同。
            </p>
            <p>
              传统的单模型方法（如全局 GARCH）假设参数在整个样本期内不变，这在存在状态切换时
              会导致严重的模型误设（Misspecification）。<strong>隐马尔可夫模型</strong>（Hidden Markov Model, HMM）
              假设存在 K 个潜在状态，每个状态有自己的统计特征，状态之间按马尔可夫链转移。
            </p>
            <p>
              识别当前市场状态对投资组合管理至关重要：低波动期可适度增加风险敞口，
              高波动期则应降低仓位、增加对冲。MarketGauge 综合仪表盘进一步将多维指标
              聚合为单一分数，提供实时状态判断。
            </p>
          </ProseBlock>
        </Section>

        {/* Section 1: HOW — 方法论 */}
        <Section index={1} title="隐马尔可夫模型（HMM）">
          <ProseBlock>
            <p>
              HMM 假设观测序列 <Formula math="y_t" /> 由潜在状态 <Formula math="s_t \in \{1, 2, ..., K\}" /> 生成。
              每个状态 k 有自己的均值 <Formula math="\mu_k" /> 和方差 <Formula math="\sigma_k^2" />：
            </p>
          </ProseBlock>

          <div className="my-6 p-4 rounded-lg bg-muted/30">
            <Formula
              block
              math={String.raw`y_t | s_t = k \sim \mathcal{N}(\mu_k, \sigma_k^2)`}
            />
          </div>

          <ProseBlock>
            <p>
              状态转移遵循马尔可夫链，转移概率矩阵 <Formula math="P" /> 的元素为：
            </p>
          </ProseBlock>

          <div className="my-6 p-4 rounded-lg bg-muted/30">
            <Formula
              block
              math={String.raw`p_{ij} = P(s_t = j | s_{t-1} = i), \quad \sum_{j=1}^K p_{ij} = 1`}
            />
          </div>

          <ProseBlock>
            <p>
              模型参数通过 <strong>Baum-Welch 算法</strong>（EM 算法的特例）估计，
              最大化观测序列的对数似然：
            </p>
          </ProseBlock>

          <div className="my-6 p-4 rounded-lg bg-muted/30">
            <Formula
              block
              math={String.raw`\mathcal{L}(\theta) = \log \sum_{s_1, ..., s_T} \prod_{t=1}^T P(y_t | s_t, \theta) P(s_t | s_{t-1})`}
            />
          </div>

          <ProseBlock>
            <p>
              选择状态数 K 时，通常通过 <strong>BIC</strong>（贝叶斯信息准则）或
              <strong>似然比检验</strong>（Likelihood Ratio Test）比较不同 K 值的模型。
              实践中 K=2（高/低波动）或 K=3（低/中/高波动）最常见。
            </p>
          </ProseBlock>
        </Section>

        {/* Section 2: WHAT — 结果展示 */}
        <Section index={2} title="HMM 状态识别结果">
          {hmm && (
            <>
              <h3 className="text-lg font-semibold text-foreground mb-3">
                状态时序图（K = {hmm.n_regimes}）
              </h3>
              <RegimeSequenceChart
                labels={hmm.labels}
                nRegimes={hmm.n_regimes}
                events={events}
                height={450}
              />
              <ReadGuide>
                <p>
                  <strong>读图要点：</strong>背景色带表示 HMM 推断的潜在状态（Regime）。
                  不同颜色对应不同状态（如绿色=低波动、黄色=中波动、红色=高波动）。
                </p>
                <p>
                  观察状态切换的时间点是否与重大事件吻合。例如，2020-01 新冠爆发、
                  2022-11 理财赎回潮期间，模型是否识别出高波动状态？
                </p>
                <p>
                  状态持续期（Duration）反映市场的"粘性"。低波动状态通常持续数月，
                  高波动状态则相对短暂但冲击强烈。
                </p>
              </ReadGuide>

              {/* Regime statistics */}
              {hmm.regime_stats.length > 0 && (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-8">
                  {hmm.regime_stats.map((stat) => (
                    <div
                      key={stat.regime}
                      className="p-4 rounded-lg border border-border/50 bg-card"
                    >
                      <div className="text-sm text-muted-foreground mb-1">
                        状态 {stat.regime + 1}
                      </div>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-xs text-muted-foreground">均值</span>
                          <span className="font-mono text-sm">{fmt(stat.mean, 4)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-xs text-muted-foreground">标准差</span>
                          <span className="font-mono text-sm">{fmt(stat.std, 4)}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </Section>

        {/* Section 3: SO WHAT — 诊断与解读 */}
        <Section index={3} title="转移矩阵与状态持续性">
          {hmm && hmm.transition_matrix.length > 0 && (
            <>
              <h3 className="text-lg font-semibold text-foreground mb-3">
                状态转移矩阵热力图
              </h3>
              <TransitionHeatmap
                matrix={hmm.transition_matrix}
                height={400}
              />
              <ReadGuide>
                <p>
                  <strong>读图要点：</strong>矩阵第 i 行第 j 列表示从状态 i 转移到状态 j 的概率
                  <Formula math="p_{ij}" />。对角线元素 <Formula math="p_{ii}" /> 表示状态持续概率。
                </p>
                <p>
                  高对角线值（如 &gt; 0.95）表示状态高度持续，切换罕见。低对角线值（如 &lt; 0.8）
                  表示状态不稳定，频繁切换。
                </p>
                <p>
                  非对角线元素反映状态间的转移方向。例如，从低波动直接跳至高波动的概率
                  <Formula math="p_{13}" /> 通常很小，市场更可能经过中波动状态逐步过渡。
                </p>
              </ReadGuide>

              {/* Transition probability metrics */}
              <div className="flex flex-wrap gap-x-6 gap-y-2 mt-6 p-4 rounded-lg bg-muted/30">
                {hmm.transition_matrix.map((row, i) => (
                  <ParamTooltip
                    key={i}
                    name={`p(${i + 1}→${i + 1})`}
                    value={row[i]}
                    decimals={4}
                    tooltip={
                      <>
                        状态 {i + 1} 的持续概率。值越接近 1，状态越稳定。
                        平均持续期 ≈ 1/(1-p) 天。
                      </>
                    }
                  />
                ))}
              </div>

              <InsightCard title="状态持续性的经济含义" variant="info">
                <p>
                  假设低波动状态的持续概率为 <Formula math="p_{11} = 0.98" />，
                  则平均持续期约为 <Formula math="1/(1-0.98) = 50" /> 天（约 2.5 个月）。
                  高波动状态的持续概率通常较低（如 0.85），平均持续期约 6-7 天，
                  说明危机往往是短暂但剧烈的冲击。
                </p>
              </InsightCard>
            </>
          )}
        </Section>

        {/* Section 4: NOW WHAT — 投资含义 */}
        <Section index={4} title="MarketGauge 仪表盘与投资建议">
          {gauge && (
            <>
              <h3 className="text-lg font-semibold text-foreground mb-3">
                综合市场状态仪表盘
              </h3>
              <MarketGaugePanel gauge={gauge} />
              <ReadGuide>
                <p>
                  <strong>读图要点：</strong>MarketGauge 将多维市场指标聚合为单一分数（0-100）。
                  分数越高，市场压力越大。状态标签（Calm/Caution/Stress/Crisis）基于分数阈值划分。
                </p>
                <p>
                  子指标（如波动率、利差、流动性）的进度条显示各维度的相对位置。
                  综合判断时需关注：(1) 分数趋势（上升/下降），(2) 各子指标是否同步恶化。
                </p>
              </ReadGuide>

              <InsightCard
                title={`当前状态：${gauge.status[1]}（${gauge.status[0]}）`}
                variant={gauge.composite < 30 ? "success" : gauge.composite < 60 ? "warning" : "info"}
              >
                <p>
                  MarketGauge 综合分数为 <strong className="font-mono">{fmt(gauge.composite, 1)}</strong>，
                  对应状态为 <strong>{gauge.status[1]}</strong>。
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

              <InsightCard title="HMM + MarketGauge 联合信号" variant="info">
                <p>
                  当 HMM 识别出高波动状态且 MarketGauge 分数 &gt; 60 时，双重信号确认市场处于压力期。
                  此时应：(1) 降低仓位至 50% 以下，(2) 增加高评级城投债占比，(3) 监控转移概率
                  <Formula math="p_{31}" />（高→低波动）以判断拐点。
                </p>
              </InsightCard>
            </>
          )}

          <ProseBlock>
            <p>
              状态识别和综合仪表盘为动态资产配置提供了量化依据。下一步将使用蒙特卡洛模拟
              生成利差的前瞻分布，评估不同情景下的组合风险。
            </p>
          </ProseBlock>
          <Link
            href="/analysis/scenarios"
            className="inline-flex items-center gap-2 mt-6 px-4 py-2 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            进入情景分析
            <ArrowRight className="h-4 w-4" />
          </Link>
        </Section>
      </div>
    </div>
  );
}
