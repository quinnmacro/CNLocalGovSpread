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
import { KalmanSignalChart } from "@/components/charts/kalman-signal";
import { ChangepointChart } from "@/components/charts/changepoint";
import { Skeleton } from "@/components/ui/skeleton";
import { useHmm, useMarketGauge, useKalmanSignal, useChangepoints } from "@/hooks/use-api";
import { fmt } from "@/lib/utils";
import Link from "next/link";
import { ArrowRight } from "lucide-react";

const PAGE_INFO = {
  title: "市场状态",
  subtitle: "Regimes — HMM + Kalman + Change Point 多维状态识别",
};

export function RegimesContent() {
  const { data: hmm, isLoading: hmmLoading, error: hmmError } = useHmm(3);
  const { data: gauge, isLoading: gaugeLoading, error: gaugeError } = useMarketGauge();
  const { data: kalman, isLoading: kalmanLoading, error: kalmanError } = useKalmanSignal();
  const { data: cpd, isLoading: cpdLoading, error: cpdError } = useChangepoints("spread_all", "binseg", 5);

  const isLoading = hmmLoading || gaugeLoading;
  const hasError = hmmError || gaugeError;

  // Key events for chart annotation
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
        <Section index={0} title="为何需要多维状态识别？">
          <ProseBlock>
            <p>
              金融时间序列往往不是平稳的（Non-stationary），而是存在<strong>结构性断裂</strong>（Structural Breaks）
              和<strong>状态切换</strong>（Regime Switching）。例如，市场在"低波动平稳期"和"高波动危机期"之间
              交替，不同状态下的收益率分布、波动率水平、相关性结构都截然不同。
            </p>
            <p>
              单一的识别方法各有侧重：<strong>隐马尔可夫模型</strong>（HMM）从波动率维度识别潜在状态；
              <strong>卡尔曼滤波</strong>（Kalman Filter）从趋势维度提取"基本面信号"与偏离度；
              <strong>变化点检测</strong>（Change Point Detection）从结构维度定位转折点。
              三种方法相互补充，形成对市场状态的立体认知。
            </p>
            <p>
              识别当前市场状态对投资组合管理至关重要：低波动期可适度增加风险敞口，
              高波动期则应降低仓位、增加对冲。MarketGauge 综合仪表盘进一步将多维指标
              聚合为单一分数，提供实时状态判断。
            </p>
          </ProseBlock>
        </Section>

        {/* Section 1: HOW — HMM 方法论 */}
        <Section index={1} title="隐马尔可夫模型（HMM）">
          <ProseBlock>
            <p>
              HMM 假设观测序列 <Formula math="y_t" /> 由潜在状态 <Formula math="s_t \in \{1, 2, ..., K\}" /> 生成。
              每个状态 k 有自己的均值 <Formula math="\mu_k" /> 和标准差 <Formula math="\sigma_k" />，
              观测概率为高斯分布：
            </p>
            <Formula
              math="y_t \mid s_t = k \sim \mathcal{N}(\mu_k, \sigma_k^2)"
              block
            />
            <p>
              状态之间的转移遵循马尔可夫性质，由转移矩阵 <Formula math="P" /> 控制：
            </p>
            <Formula
              math="p_{ij} = P(s_t = j \mid s_{t-1} = i), \quad \sum_j p_{ij} = 1"
              block
            />
            <p>
              模型通过 Baum-Welch（EM）算法估计参数，然后用 Viterbi 算法解码最可能的状态序列。
              这里使用 <Formula math="K = 3" /> 个状态，对应低、中、高波动区间。
            </p>
          </ProseBlock>
        </Section>

        {/* Section 1.5: HOW — Kalman Filter 方法论 */}
        <Section title="卡尔曼滤波（Kalman Filter）">
          <ProseBlock>
            <p>
              卡尔曼滤波将利差分解为不可观测的<strong>趋势（信号）</strong>和<strong>短期噪声</strong>。
              这与有效市场假说一致：利差围绕"基本面价值"波动，短期偏离最终会均值回归。
            </p>
            <p>使用 <strong>Local Level Model</strong>（局部水平模型）：</p>
            <Formula
              math="\begin{aligned} y_t &= \mu_t + \varepsilon_t, \quad &\varepsilon_t \sim \mathcal{N}(0, \sigma^2_\varepsilon) \quad &\text{（观测方程）} \\ \mu_{t+1} &= \mu_t + \eta_t, \quad &\eta_t \sim \mathcal{N}(0, \sigma^2_\eta) \quad &\text{（状态方程）} \end{aligned}"
              block
            />
            <p>
              信噪比 <Formula math="Q = \sigma^2_\eta / \sigma^2_\varepsilon" /> 决定了滤波的平滑度：
            </p>
            <div className="flex flex-wrap gap-x-6 gap-y-2 mt-3 p-4 rounded-lg bg-muted/30">
              <ParamTooltip
                name="Q → 0"
                value="信号几乎不变"
                tooltip={<>状态方差极小，Kalman 滤波器近似于全局均值，利差强均值回归。</>}
              />
              <ParamTooltip
                name="Q → ∞"
                value="信号 ≈ 原始数据"
                tooltip={<>状态方差远大于观测噪声，滤波器几乎不降噪，信号跟踪每一次波动。</>}
              />
            </div>
            <p>
              偏离度（Deviation）定义为 <Formula math="d_t = y_t - \mu_t" />，
              标准化后得到 z-score：<Formula math="z_t = d_t / \text{std}_{60}(d)" />。
              当 <Formula math="|z_t| > 1.5" /> 时，认为利差显著偏离基本面趋势。
            </p>
          </ProseBlock>
        </Section>

        {/* Section 1.6: HOW — Change Point Detection 方法论 */}
        <Section title="结构性变化点检测（Change Point Detection）">
          <ProseBlock>
            <p>
              变化点检测与 HMM 互补：HMM 识别"当前处于什么状态"，CPD 定位"状态何时发生变化"。
              使用 <strong>Binary Segmentation</strong>（二分法）配合 <Formula math="\ell_2" /> 损失函数：
            </p>
            <Formula
              math="\min_{\tau_1, \ldots, \tau_K} \sum_{k=0}^{K} \sum_{t=\tau_k}^{\tau_{k+1}-1} (y_t - \bar{y}_{[\tau_k, \tau_{k+1})})^2"
              block
            />
            <p>
              算法递归地将序列分为子段，每次选择使总残差平方和下降最大的分裂点，
              直到达到预设的分段数 <Formula math="K" />。这比 PELT（Pruned Exact Linear Time）
              更适合控制宏观层面的结构识别。
            </p>
            <p>
              每个分段有自己的均值 <Formula math="\bar{y}_k" /> 和标准差，
              可用于判断利差在不同时期的"运行区间"。
            </p>
          </ProseBlock>
        </Section>

        {/* Section 2: WHAT — HMM 结果 */}
        <Section index={2} title="HMM 状态序列与转移矩阵">
          {hmm && (
            <>
              <h3 className="text-lg font-semibold text-foreground mb-3">
                波动率状态时序（Viterbi 解码）
              </h3>
              <RegimeSequenceChart
                labels={hmm.labels}
                nRegimes={hmm.n_regimes}
                events={events}
              />
              <ReadGuide>
                <p>
                  <strong>读图要点：</strong>背景色表示 Viterbi 解码的潜在状态——
                  绿色为低波动（稳定期），蓝色为中波动（过渡期），红色为高波动（危机期）。
                  箭头标注重要事件发生的时间点，观察状态切换是否与事件吻合。
                </p>
              </ReadGuide>
            </>
          )}

          {hmm && (
            <>
              <h3 className="text-lg font-semibold text-foreground mt-8 mb-3">
                状态转移概率矩阵
              </h3>
              <TransitionHeatmap matrix={hmm.transition_matrix} />
              <ReadGuide>
                <p>
                  <strong>读图要点：</strong>对角线元素 <Formula math="p_{ii}" /> 表示状态持续概率。
                  颜色越深（越接近 1），状态越"持久"。右上角 <Formula math="p_{13}" /> 表示
                  从低波动直接跳到高波动的概率——通常很小，说明市场变化多为渐进式。
                </p>
              </ReadGuide>

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

        {/* Section 2.5: WHAT — Kalman Signal 结果 */}
        <Section title="Kalman 滤波信号提取">
          {kalmanLoading && <Skeleton className="h-96 w-full" />}
          {kalmanError && (
            <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive">
              Kalman 信号数据加载失败
            </div>
          )}
          {kalman && (
            <>
              <h3 className="text-lg font-semibold text-foreground mb-3">
                趋势信号与偏离度
              </h3>
              <KalmanSignalChart
                spread={kalman.signal.map((s, i) => ({
                  date: s.date,
                  value: kalman.deviation[i]?.value != null
                    ? s.value + kalman.deviation[i].value
                    : s.value,
                }))}
                signal={kalman.signal}
                deviationZscore={kalman.deviation_zscore}
                events={events}
              />
              <ReadGuide>
                <p>
                  <strong>上方面板：</strong>灰色线为原始利差，橙色线为 Kalman 滤波提取的平滑趋势（"基本面信号"）。
                  两者的差异即短期噪声，反映市场情绪等非基本面驱动。
                </p>
                <p>
                  <strong>下方面板：</strong>偏离度 z-score 衡量利差偏离趋势的程度。
                  z &gt; 1.5（红色背景）表示利差异常偏高（走阔），z &lt; −1.5（绿色背景）表示利差异常偏低（收窄）。
                </p>
              </ReadGuide>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
                <MetricCard
                  label="信号强度"
                  value={fmt(kalman.signal_strength, 3)}
                  unit="sigmoid(|z|)"
                />
                <MetricCard
                  label="当前状态"
                  value={
                    kalman.is_overvalued
                      ? "高估"
                      : kalman.is_undervalued
                        ? "低估"
                        : "合理"
                  }
                  unit="|z| > 1.5 ?"
                />
                <MetricCard
                  label="信噪比 Q"
                  value={fmt(kalman.q_ratio, 6)}
                  unit="σ²_η / σ²_ε"
                />
                <MetricCard
                  label="观测噪声 σ²_ε"
                  value={fmt(kalman.sigma2_eps, 4)}
                  unit="短期波动方差"
                />
              </div>

              <div className="flex flex-wrap gap-x-6 gap-y-2 mt-6 p-4 rounded-lg bg-muted/30">
                <ParamTooltip
                  name="σ²_η"
                  value={kalman.sigma2_eta}
                  decimals={6}
                  tooltip={
                    <>
                      状态转移方差：趋势本身的波动性。越大说明基本面趋势变化越频繁。
                      对于城投债利差，该值通常很小，反映趋势缓慢变化。
                    </>
                  }
                />
                <ParamTooltip
                  name="σ²_ε"
                  value={kalman.sigma2_eps}
                  decimals={4}
                  tooltip={
                    <>
                      观测噪声方差：利差围绕趋势的短期波动。
                      包含市场情绪、流动性冲击等非基本面因素。
                    </>
                  }
                />
                <ParamTooltip
                  name="Q"
                  value={kalman.q_ratio}
                  decimals={6}
                  tooltip={
                    <>
                      信噪比 = σ²_η / σ²_ε。Q 越小，滤波器越平滑（更强调趋势）；
                      Q 越大，信号越贴近原始数据（更强调短期变化）。
                    </>
                  }
                />
              </div>

              <InsightCard
                title={`Kalman 信号判读：${kalman.is_overvalued ? "利差高估" : kalman.is_undervalued ? "利差低估" : "利差处于合理区间"}`}
                variant={kalman.is_overvalued ? "warning" : kalman.is_undervalued ? "success" : "info"}
              >
                <p>
                  当前信号强度为 <strong className="font-mono">{fmt(kalman.signal_strength, 3)}</strong>，
                  信噪比 <Formula math={`Q = ${fmt(kalman.q_ratio, 6)}`} />。
                  {kalman.q_ratio < 0.01 && (
                    <> 极小的 Q 值说明利差趋势非常平滑，短期波动远大于趋势变化。</>
                  )}
                  {kalman.is_overvalued && (
                    <> 利差显著高于趋势（z &gt; 1.5），可能面临收窄压力。</>
                  )}
                  {kalman.is_undervalued && (
                    <> 利差显著低于趋势（z &lt; −1.5），可能面临走阔压力。</>
                  )}
                </p>
              </InsightCard>
            </>
          )}
        </Section>

        {/* Section 2.6: WHAT — Change Point Detection 结果 */}
        <Section title="结构性变化点">
          {cpdLoading && <Skeleton className="h-80 w-full" />}
          {cpdError && (
            <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive">
              变化点检测数据加载失败
            </div>
          )}
          {cpd && (
            <>
              <h3 className="text-lg font-semibold text-foreground mb-3">
                Binary Segmentation 检测结果（{cpd.n_segments} 个分段）
              </h3>
              <ChangepointChart
                spread={kalman?.signal.map((s, i) => ({
                  date: s.date,
                  value: kalman.deviation[i]?.value != null
                    ? s.value + kalman.deviation[i].value
                    : s.value,
                })) ?? []}
                breakpointDates={cpd.breakpoint_dates}
                segments={cpd.segments}
                events={events}
              />
              <ReadGuide>
                <p>
                  <strong>读图要点：</strong>彩色粗线表示每个分段内的均值水平，垂直虚线标记结构变化点。
                  当虚线与重要事件时间点吻合时，说明该事件确实导致了利差的结构性变化。
                </p>
                <p>
                  对比 HMM 的状态切换和 CPD 的断裂点：HMM 是概率性的、可以来回切换；
                  CPD 是确定性的、表示均值的永久改变。两者结合可以区分"暂时波动"与"趋势转变"。
                </p>
              </ReadGuide>

              {/* Segment table */}
              <div className="mt-6 overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="border-b border-border/50">
                      <th className="text-left py-2 px-3 text-muted-foreground font-medium">段</th>
                      <th className="text-left py-2 px-3 text-muted-foreground font-medium">起始日期</th>
                      <th className="text-left py-2 px-3 text-muted-foreground font-medium">结束日期</th>
                      <th className="text-right py-2 px-3 text-muted-foreground font-medium">均值 (bps)</th>
                      <th className="text-right py-2 px-3 text-muted-foreground font-medium">标准差</th>
                      <th className="text-right py-2 px-3 text-muted-foreground font-medium">天数</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cpd.segments.map((seg, i) => (
                      <tr key={i} className="border-b border-border/20 hover:bg-muted/20">
                        <td className="py-1.5 px-3 font-mono">{i + 1}</td>
                        <td className="py-1.5 px-3 font-mono text-xs">{seg.start_date}</td>
                        <td className="py-1.5 px-3 font-mono text-xs">{seg.end_date}</td>
                        <td className="py-1.5 px-3 font-mono text-right">{fmt(seg.mean)}</td>
                        <td className="py-1.5 px-3 font-mono text-right">{fmt(seg.std)}</td>
                        <td className="py-1.5 px-3 font-mono text-right">{seg.end_idx - seg.start_idx}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <InsightCard title="CPD vs HMM：互补视角" variant="info">
                <p>
                  HMM 假设状态可以反复切换（马尔可夫链），适合描述"周期性波动"；
                  CPD 假设每个断裂点之后进入一个全新的均值水平，适合识别"趋势性变化"。
                  例如，如果 CPD 检测到利差从 30 bps 跳升至 80 bps 且不再回落，
                  这是一个<strong>结构性改变</strong>（可能对应政策变化或信用事件），
                  而非 HMM 所描述的"暂时进入高波动状态"。
                </p>
              </InsightCard>
            </>
          )}
        </Section>

        {/* Section 3: SO WHAT — 诊断与交叉验证 */}
        <Section index={3} title="多方法交叉验证">
          <ProseBlock>
            <p>
              三种方法从不同维度刻画市场状态：HMM 关注<strong>波动率水平</strong>，
              Kalman 关注<strong>趋势偏离度</strong>，CPD 关注<strong>均值结构变化</strong>。
              交叉验证可以增强信号的可信度。
            </p>
          </ProseBlock>

          {hmm && kalman && (
            <InsightCard title="HMM × Kalman 联合判读" variant="info">
              <p>
                当前 HMM 状态为 <strong>{hmm.current_regime_name}</strong>，
                Kalman 判读为 <strong>
                  {kalman.is_overvalued ? "高估" : kalman.is_undervalued ? "低估" : "合理"}
                </strong>。
              </p>
              <ul className="list-disc list-inside mt-3 space-y-1">
                {hmm.current_regime >= 2 && kalman.is_overvalued && (
                  <li>双重警告：高波动状态 + 利差高估，市场可能正在经历压力释放</li>
                )}
                {hmm.current_regime === 0 && kalman.is_undervalued && (
                  <li>低波动 + 利差低估：可能是配置窗口，但需警惕波动率回升</li>
                )}
                {hmm.current_regime === 0 && !kalman.is_overvalued && !kalman.is_undervalued && (
                  <li>低波动 + 利差合理：市场平稳，适合持有策略</li>
                )}
              </ul>
            </InsightCard>
          )}

          {cpd && (
            <InsightCard title="CPD 时间线解读" variant="info">
              <p>
                共检测到 <strong>{cpd.n_segments}</strong> 个结构性分段，
                最后一个变化点发生在 <strong>{cpd.breakpoint_dates[cpd.breakpoint_dates.length - 1] ?? "N/A"}</strong>。
                {cpd.segments.length >= 2 && (() => {
                  const last = cpd.segments[cpd.segments.length - 1];
                  const prev = cpd.segments[cpd.segments.length - 2];
                  const direction = last.mean > prev.mean ? "上升" : "下降";
                  return (
                    <> 当前段均值 {fmt(last.mean)} bps，较前一段（{fmt(prev.mean)} bps）{direction} {fmt(Math.abs(last.mean - prev.mean))} bps。</>
                  );
                })()}
              </p>
            </InsightCard>
          )}
        </Section>

        {/* Section 4: NOW WHAT — MarketGauge + 投资建议 */}
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

              <InsightCard title="HMM + MarketGauge + Kalman 联合信号" variant="info">
                <p>
                  当 HMM 识别出高波动状态且 MarketGauge 分数 &gt; 60 且 Kalman z-score &gt; 1.5 时，
                  三重信号确认市场处于压力期。此时应：
                  (1) 降低仓位至 50% 以下，(2) 增加高评级城投债占比，
                  (3) 监控转移概率 <Formula math="p_{31}" />（高→低波动）以判断拐点，
                  (4) 等待 Kalman z-score 回归至 ±1.0 以内再考虑加仓。
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
