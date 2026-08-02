"use client";
import { useState, useEffect, useMemo } from "react";

import { Sidebar } from "@/components/layout/sidebar";
import { Breadcrumb } from "@/components/layout/breadcrumb";
import { Section } from "@/components/narrative/section";
import { ProseBlock } from "@/components/narrative/prose-block";
import { Formula } from "@/components/narrative/formula";
import { ReadGuide } from "@/components/narrative/read-guide";
import { InsightCard } from "@/components/narrative/insight-card";
import { ParamTooltip } from "@/components/narrative/param-tooltip";
import { ChartWrapper } from "@/components/narrative/chart-wrapper";
import { VolatilityOverlay } from "@/components/charts/volatility-overlay";
import { TournamentTable } from "@/components/charts/tournament-table";
import { ResidualDiagnostics } from "@/components/charts/residual-diagnostics";
import { ModelSelector } from "@/components/interactive/model-selector";
import { HARRVDecomposition } from "@/components/charts/har-rv-decomposition";
import { StochasticVolBand } from "@/components/charts/stochastic-vol-band";
import { GASScoreDynamics } from "@/components/charts/gas-score-dynamics";
import { MSGARCHRegimes } from "@/components/charts/ms-garch-regimes";
import { VolatilityModelComparison } from "@/components/charts/volatility-model-comparison";
import { Skeleton } from "@/components/ui/skeleton";
import { PageNavigation } from "@/components/narrative/page-navigation";
import {
  useTournament,
  useModelDetail,
  useFigarch,
  useFitCustom,
  useHarRv,
  useStochasticVol,
  useGas,
  useMsGarch,
} from "@/hooks/use-api";
import { useScrollSpy } from "@/hooks/use-scroll-spy";
import { fmt } from "@/lib/utils";
import type { TimePoint, CustomFitRequest } from "@/lib/types";
import Link from "next/link";
import {
  ArrowRight,
  ChevronUp,
  Activity,
  BarChart3,
  TrendingUp,
  Layers,
  Brain,
  Target,
  Zap,
} from "lucide-react";

const PAGE_INFO = {
  title: "波动率建模",
  subtitle: "Volatility — 经典 GARCH 到前沿 HAR-RV / SV / GAS / MS-GARCH 全谱系",
};

const TOC_SECTIONS = [
  { id: "why", label: "WHY", sublabel: "研究动机" },
  { id: "how", label: "HOW", sublabel: "方法论" },
  { id: "what", label: "WHAT", sublabel: "结果展示" },
  { id: "so-what", label: "SO WHAT", sublabel: "诊断" },
  { id: "now-what", label: "NOW WHAT", sublabel: "投资" },
] as const;

const SECTION_IDS = TOC_SECTIONS.map((s) => s.id) as unknown as readonly string[];

export function VolatilityContent() {
  // Classic models
  const { data: tournament, isLoading: tournamentLoading, error: tournamentError } = useTournament();
  const { data: figarch, isLoading: figarchLoading } = useFigarch();
  const fitCustom = useFitCustom();

  // Modern models
  const { data: harRv, isLoading: harLoading } = useHarRv();
  const { data: sv, isLoading: svLoading } = useStochasticVol();
  const { data: gas, isLoading: gasLoading } = useGas();
  const { data: msGarch, isLoading: msLoading } = useMsGarch();

  const isLoading = tournamentLoading || figarchLoading || harLoading || gasLoading || msLoading;
  const activeSection = useScrollSpy(SECTION_IDS, 120, !isLoading);

  const [showBackToTop, setShowBackToTop] = useState(false);
  useEffect(() => {
    const onScroll = () => setShowBackToTop(window.scrollY > 400);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Default model selector
  const defaultModel =
    tournament?.winner_aic?.toLowerCase() ??
    tournament?.models[0]?.model_name?.toLowerCase() ??
    "garch";
  const [selectedModel, setSelectedModel] = useState<string>(defaultModel);

  useEffect(() => {
    if (tournament?.winner_aic && selectedModel === defaultModel) {
      setSelectedModel(tournament.winner_aic);
    }
  }, [tournament, selectedModel, defaultModel]);

  const { data: modelDetail, isLoading: detailLoading } = useModelDetail(
    selectedModel?.toLowerCase(),
    {
      enabled:
        !!selectedModel &&
        /^(garch|egarch|gjr|ewma|figarch)$/.test(selectedModel.toLowerCase()),
    }
  );

  const hasError = tournamentError;
  const residuals = modelDetail?.standardized_residuals ?? [];
  const diagnostics = modelDetail?.diagnostics ?? null;

  function handleFitCustom(request: CustomFitRequest) {
    fitCustom.mutate(request, {
      onSuccess: (result) => setSelectedModel(result.model_name),
    });
  }

  // Build comparison data
  const allModelsForComparison = useMemo(() => {
    const models: { name: string; volatility: TimePoint[] }[] = [];
    if (modelDetail?.conditional_volatility?.length) {
      models.push({ name: modelDetail.model_name, volatility: modelDetail.conditional_volatility });
    }
    if (figarch?.conditional_volatility?.length) {
      models.push({ name: figarch.model_name, volatility: figarch.conditional_volatility });
    }
    if (harRv?.conditional_volatility?.length) {
      models.push({ name: harRv.model_name, volatility: harRv.conditional_volatility });
    }
    if (sv?.conditional_volatility?.length) {
      models.push({ name: sv.model_name, volatility: sv.conditional_volatility });
    }
    if (gas?.conditional_volatility?.length) {
      models.push({ name: gas.model_name, volatility: gas.conditional_volatility });
    }
    if (msGarch?.conditional_volatility?.length) {
      models.push({ name: msGarch.model_name, volatility: msGarch.conditional_volatility });
    }
    return models;
  }, [modelDetail, figarch, harRv, sv, gas, msGarch]);

  // ── Loading / Error ────────────────────────────────────────

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
            <Skeleton className="h-64 w-full" />
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
            <p className="text-destructive font-medium">模型数据加载失败</p>
            <p className="text-sm text-muted-foreground mt-2">
              请检查后端 API 连接或稍后重试
            </p>
          </div>
        </div>
      </div>
    );
  }

  // ── Main Render ────────────────────────────────────────────

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

        {/* ── Section 0: WHY ────────────────────────────────── */}
        <Section id="why" index={0} title="WHY — 为什么需要波动率建模">
          <ProseBlock>
            <p>
              城投债利差的波动率不是常数——它随时间聚集、跳跃、均值回归。
              传统 GARCH(1,1) 假设单一的指数衰减记忆结构，无法解释现实中
              日/周/月不同时间尺度交易者带来的<strong>异质记忆</strong>，
              也无法捕捉波动率本身的<strong>随机性</strong>（SV）和
              <strong>状态切换</strong>（MS-GARCH）。
            </p>
            <p className="mt-4">
              本页将 6 种波动率方法分为两大阵营：
            </p>
            <ul className="list-disc list-inside mt-2 space-y-1">
              <li>
                <strong>经典族</strong>：GARCH(1,1)、EGARCH、GJR-GARCH、EWMA、FIGARCH
                — 教科书级别的确定性条件方差模型
              </li>
              <li>
                <strong>前沿族</strong>：HAR-RV（异质自回归）、Stochastic Vol（贝叶斯随机波动率）、
                GAS（Score 驱动）、MS-GARCH（状态切换）
                — 2020s 前沿实践
              </li>
            </ul>
          </ProseBlock>

          <InsightCard title="核心差异：条件方差 vs 潜在过程" variant="info">
            <p>
              GARCH 类模型假设条件方差 <Formula math="\sigma_t^2" /> 是
              过去信息的<strong>确定性函数</strong>。SV 模型则认为波动率
              是独立的<strong>随机过程</strong>，有自己的驱动噪声。
              HAR-RV 用三个窗口捕捉异质交易者的不同记忆，
              GAS 用 score 函数替代固定规则实现稳健更新。
            </p>
          </InsightCard>
        </Section>

        <div className="section-divider" />

        {/* ── Section 1: HOW ────────────────────────────────── */}
        <Section id="how" index={1} title="HOW — 方法论与数学基础">
          {/* GARCH */}
          <ProseBlock>
            <h3 className="text-lg font-semibold text-foreground mb-3">
              1. GARCH(1,1) — Bollerslev 1986
            </h3>
            <p>
              最经典的条件方差模型。方差由过去残差平方（ARCH 项）和过去方差（GARCH 项）线性组合：
            </p>
            <Formula
              math="\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2"
              block
            />
            <p>
              持续性参数 <Formula math="\alpha + \beta" /> 越接近 1，
              冲击衰减越慢。若 <Formula math="\alpha + \beta = 1" />，退化为 IGARCH（单位根）。
            </p>
          </ProseBlock>

          {/* EGARCH */}
          <ProseBlock>
            <h3 className="text-lg font-semibold text-foreground mb-3">
              2. EGARCH — Nelson 1991
            </h3>
            <p>
              对数方差建模，天然保证方差为正，且可捕捉<strong>杠杆效应</strong>（负冲击放大波动率）：
            </p>
            <Formula
              math="\log\sigma_t^2 = \omega + \alpha\left(\left|\frac{\varepsilon_{t-1}}{\sigma_{t-1}}\right| - \sqrt{2/\pi}\right) + \gamma \frac{\varepsilon_{t-1}}{\sigma_{t-1}} + \beta \log\sigma_{t-1}^2"
              block
            />
            <p>
              <Formula math="\gamma < 0" /> 表示负冲击（利差走阔）对波动率影响更大。
            </p>
          </ProseBlock>

          {/* EWMA */}
          <ProseBlock>
            <h3 className="text-lg font-semibold text-foreground mb-3">
              3. EWMA — RiskMetrics 1994
            </h3>
            <p>
              IGARCH 特例，<Formula math="\alpha + \beta = 1" />，只有一个参数 <Formula math="\lambda" />：
            </p>
            <Formula
              math="\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda) r_{t-1}^2"
              block
            />
            <p>
              无均值回归，预测为常数。优点：极简、无需估计均值方程。
              λ 通过 QLIKE 损失自动校准。
            </p>
          </ProseBlock>

          {/* FIGARCH */}
          <ProseBlock>
            <h3 className="text-lg font-semibold text-foreground mb-3">
              4. FIGARCH — Baillie 1996
            </h3>
            <p>
              分数积分 GARCH，用分数差分参数 <Formula math="d \in (0,1)" /> 捕捉<strong>长记忆</strong>：
            </p>
            <Formula
              math="(1-\beta L)\sigma_t^2 = \omega + [1 - \beta L - (1-\phi L)(1-L)^d]\varepsilon_t^2"
              block
            />
            <p>
              当 <Formula math="d > 0" /> 时，冲击以双曲速率衰减（比 GARCH 的指数衰减慢得多）。
            </p>
          </ProseBlock>

          {/* HAR-RV */}
          <ProseBlock>
            <h3 className="text-lg font-semibold text-foreground mb-3">
              5. HAR-RV — Corsi 2009 ⭐
            </h3>
            <p>
              用三个滞后窗口（日/周/月）分别捕捉不同时间尺度交易者的波动率贡献：
            </p>
            <Formula
              math="RV_t = \beta_0 + \beta_d \cdot RV_{t-1} + \beta_w \cdot RV_{t-1}^{(w)} + \beta_m \cdot RV_{t-1}^{(m)} + \epsilon_t"
              block
            />
            <p>
              其中 <Formula math="RV^{(w)}" /> 是 5 日滚动均值，<Formula math="RV^{(m)}" /> 是 22 日滚动均值。
              比 GARCH 的单一记忆结构更符合市场微观结构——日交易者、周交易者和月度投资者各有不同的信息衰减速度。
            </p>
          </ProseBlock>

          {/* SV */}
          <ProseBlock>
            <h3 className="text-lg font-semibold text-foreground mb-3">
              6. Stochastic Volatility — Taylor 1986 / Kim-Shephard-Chib 1998 ⭐
            </h3>
            <p>
              波动率是独立的随机过程，而非过去信息的确定性函数：
            </p>
            <Formula
              math="r_t = \exp(h_t/2) \cdot \varepsilon_t, \quad h_t = \mu + \phi(h_{t-1} - \mu) + \sigma_\eta \eta_t"
              block
            />
            <p>
              贝叶斯框架（PyMC + ADVI）给出参数的完整后验分布——不只是
              <Formula math="\sigma = 0.02" />，而是
              <Formula math="\sigma \in [0.015, 0.028]" />（80% 概率）。
            </p>
          </ProseBlock>

          {/* GAS */}
          <ProseBlock>
            <h3 className="text-lg font-semibold text-foreground mb-3">
              7. GAS(1,1) — Creal, Koopman, Lucas 2013 ⭐
            </h3>
            <p>
              GARCH 的理论泛化。用 score 函数（对数似然的梯度）驱动时变参数更新：
            </p>
            <Formula
              math="f_{t+1} = \omega + A \cdot s_t + B \cdot f_t"
              block
            />
            <p>
              正态分布时退化为 GARCH；Student-t 分布时，score 自动对异常值降权——
              不会因为一个极端收益率就大幅抬升波动率预测，比 GARCH 更稳健。
            </p>
          </ProseBlock>

          {/* MS-GARCH */}
          <ProseBlock>
            <h3 className="text-lg font-semibold text-foreground mb-3">
              8. MS-GARCH — Markov-Switching GARCH ⭐
            </h3>
            <p>
              波动率参数随隐藏状态（regime）切换。桥接 regimes 页面的 HMM 状态检测：
            </p>
            <Formula
              math="\sigma_{k,t}^2 = \omega_k + \alpha_k r_{t-1}^2 + \beta_k \sigma_{k,t-1}^2, \quad S_t \sim \text{Markov}(\Pi)"
              block
            />
            <p>
              平静期用低持久性参数，危机期用高持久性参数。
              解释了为什么波动率聚集不是"一次冲击慢慢衰减"，而是"不同状态之间的切换"。
            </p>
          </ProseBlock>
        </Section>

        <div className="section-divider" />

        {/* ── Section 2: WHAT ────────────────────────────────── */}
        <Section id="what" index={2} title="WHAT — 拟合结果与可视化">
          {/* Tournament */}
          <h3 className="text-lg font-semibold text-foreground mb-3">
            模型锦标赛（Model Tournament）
          </h3>
          {tournament && (
            <>
              <ChartWrapper title="模型锦标赛">
                <TournamentTable
                  models={tournament.models}
                  winnerAic={tournament.winner_aic}
                  winnerBic={tournament.winner_bic}
                  onModelClick={(name) => setSelectedModel(name)}
                  selectedModel={selectedModel}
                />
              </ChartWrapper>
              <ReadGuide>
                <p>
                  <strong>读表要点：</strong>AIC/BIC 越低越好（在拟合优度和复杂度之间权衡）。
                  "持续性"列显示 <Formula math="\alpha + \beta" />：越接近 1，波动率记忆越长。
                  ARCH 列显示残差是否仍有条件异方差效应，正态列显示残差是否近似正态。
                </p>
              </ReadGuide>
            </>
          )}

          {/* HAR-RV */}
          {harRv && (
            <>
              <h3 className="text-lg font-semibold text-foreground mt-8 mb-3">
                HAR-RV 三窗口分解
              </h3>
              <ChartWrapper title="HAR-RV 三窗口分解">
                <HARRVDecomposition
                  rvDaily={harRv.rv_daily}
                  rvWeekly={harRv.rv_weekly}
                  rvMonthly={harRv.rv_monthly}
                  condVol={harRv.conditional_volatility}
                />
              </ChartWrapper>
              <ReadGuide>
                <p>
                  <strong>读图要点：</strong>日线（浅蓝）高频噪声大，周线（橙）和月线（绿）
                  依次平滑。虚线是 HAR 模型的拟合方差——三条线的加权组合。
                  R² = {fmt(harRv.r_squared, 4)} 表示三个窗口解释了多少已实现方差的变异。
                </p>
                <p>
                  <Formula math="\beta_d" /> 反映日内交易者的影响权重，
                  <Formula math="\beta_w" /> 反映周度趋势交易者的影响，
                  <Formula math="\beta_m" /> 反映月度宏观投资者的影响。
                </p>
              </ReadGuide>
              {harRv.params && (
                <div className="mt-4 space-y-2">
                  <p>
                    <ParamTooltip
                      name="β_d (日)"
                      value={harRv.params.beta_d}
                      tooltip="日内交易者的波动率传导权重，反映短期信息的衰减速度。"
                    />
                  </p>
                  <p>
                    <ParamTooltip
                      name="β_w (周)"
                      value={harRv.params.beta_w}
                      tooltip="周度趋势交易者的传导权重，捕捉 5 日尺度的波动率成分。"
                    />
                  </p>
                  <p>
                    <ParamTooltip
                      name="β_m (月)"
                      value={harRv.params.beta_m}
                      tooltip="月度投资者的传导权重，捕捉 22 日尺度的长记忆成分。"
                    />
                  </p>
                </div>
              )}
            </>
          )}

          {/* Stochastic Volatility */}
          {sv && (
            <>
              <h3 className="text-lg font-semibold text-foreground mt-8 mb-3">
                随机波动率后验区间
              </h3>
              <ChartWrapper title="随机波动率后验区间">
                <StochasticVolBand
                  condVol={sv.conditional_volatility}
                  volLower={sv.vol_lower}
                  volUpper={sv.vol_upper}
                />
              </ChartWrapper>
              <ReadGuide>
                <p>
                  <strong>读图要点：</strong>蓝线是后验均值波动率，填充区域是 80% 可信区间。
                  区间越宽，波动率估计的不确定性越大。SV 的核心价值在于量化"波动率的波动率"——
                  不只告诉你当前 σ 是多少，还告诉你这个估计有多可靠。
                </p>
              </ReadGuide>
              {sv.params && (
                <div className="mt-4 space-y-2">
                  <p>
                    <ParamTooltip
                      name="μ (长期水平)"
                      value={sv.params.mu}
                      tooltip="对数波动率的长期均值。exp(μ/2) 近似等于长期波动率水平。"
                    />
                  </p>
                  <p>
                    <ParamTooltip
                      name="φ (持续性)"
                      value={sv.params.phi}
                      tooltip="对数波动率的 AR(1) 系数。φ 接近 1 表示波动率变化非常缓慢。"
                    />
                  </p>
                  <p>
                    <ParamTooltip
                      name="σ_η (波动率的波动率)"
                      value={sv.params.sigma_eta}
                      tooltip="波动率过程自身的创新标准差。σ_η 越大，波动率越不可预测。"
                    />
                  </p>
                </div>
              )}
            </>
          )}

          {/* GAS */}
          {gas && (
            <>
              <h3 className="text-lg font-semibold text-foreground mt-8 mb-3">
                GAS Score 动态
              </h3>
              <ChartWrapper title="GAS Score 动态">
                <GASScoreDynamics
                  condVol={gas.conditional_volatility}
                  scoreSeries={gas.score_series}
                />
              </ChartWrapper>
              <ReadGuide>
                <p>
                  <strong>读图要点：</strong>上面板是 GAS 条件波动率，下面板是 score 更新量
                  （绿色=正向更新，红色=负向更新）。Score 是驱动波动率变化的"引擎"。
                  正态分布时 score ∝ (r² - σ²)，Student-t 时 score 对极端值自动降权。
                </p>
              </ReadGuide>
              {gas.params && (
                <div className="mt-4 space-y-2">
                  {gas.params.A != null && (
                    <p>
                      <ParamTooltip
                        name="A (Score 系数)"
                        value={gas.params.A}
                        tooltip="Score 的放大系数。A 越大，新信息对波动率预测的影响越强。"
                      />
                    </p>
                  )}
                  {gas.params.B != null && (
                    <p>
                      <ParamTooltip
                        name="B (持续性)"
                        value={gas.params.B}
                        tooltip="波动率的自回归系数，类似 GARCH 的 β。B 接近 1 表示高持续性。"
                      />
                    </p>
                  )}
                  {gas.params.nu != null && (
                    <p>
                      <ParamTooltip
                        name="ν (自由度)"
                        value={gas.params.nu}
                        tooltip="Student-t 分布自由度。ν 越小尾部越厚，ν→∞ 退化为正态。"
                      />
                    </p>
                  )}
                </div>
              )}
            </>
          )}

          {/* MS-GARCH */}
          {msGarch && (
            <>
              <h3 className="text-lg font-semibold text-foreground mt-8 mb-3">
                MS-GARCH 状态切换
              </h3>
              <ChartWrapper title="MS-GARCH 状态切换">
                <MSGARCHRegimes
                  condVol={msGarch.conditional_volatility}
                  regimeLabels={msGarch.regime_labels}
                  regimeParams={msGarch.regime_params}
                />
              </ChartWrapper>
              <ReadGuide>
                <p>
                  <strong>读图要点：</strong>散点按 HMM 识别的波动率状态着色
                  （绿=低波，黄=高波，红=极端）。不同状态下 GARCH 参数完全不同——
                  这解释了为什么单一 GARCH 的"持续性"估计常常偏高：
                  实际上不是单次冲击衰减慢，而是在高波和低波状态之间切换。
                </p>
              </ReadGuide>
              <InsightCard
                title={`当前状态：${msGarch.current_regime_name}（Regime ${msGarch.current_regime}）`}
                variant={msGarch.current_regime === 0 ? "success" : msGarch.current_regime === 1 ? "warning" : "info"}
              >
                <p>
                  MS-GARCH 当前识别市场处于 <strong>{msGarch.current_regime_name}</strong> 状态。
                </p>
                {msGarch.regime_params.map((rp) => (
                  <p key={rp.regime} className="mt-1 text-sm">
                    Regime {rp.regime}: ω={fmt(rp.omega, 6)},
                    α={fmt(rp.alpha, 4)}, β={fmt(rp.beta, 4)},
                    持续性={fmt(rp.persistence, 4)}
                  </p>
                ))}
              </InsightCard>
            </>
          )}

          {/* Multi-model comparison */}
          {allModelsForComparison.length > 1 && (
            <>
              <h3 className="text-lg font-semibold text-foreground mt-8 mb-3">
                全模型条件波动率对比
              </h3>
              <ChartWrapper title="全模型条件波动率对比">
                <VolatilityModelComparison models={allModelsForComparison} />
              </ChartWrapper>
              <ReadGuide>
                <p>
                  <strong>读图要点：</strong>所有模型的条件波动率在大趋势上一致，
                  但在细节上有差异——GARCH 的衰减更平滑，HAR-RV 保留了更多高频结构，
                  SV 提供了不确定性区间，MS-GARCH 在状态切换处有阶跃式变化。
                  选择哪个模型取决于你的应用：风险度量偏好保守估计（SV 上界），
                  交易信号偏好灵敏估计（HAR-RV）。
                </p>
              </ReadGuide>
            </>
          )}

          {/* Classic model detail */}
          {modelDetail && (
            <>
              <h3 className="text-lg font-semibold text-foreground mt-8 mb-3">
                经典模型详情：{modelDetail.model_name}
              </h3>
              <ModelSelector
                onFit={handleFitCustom}
                isFitting={fitCustom.isPending}
              />
              <ChartWrapper title={`经典模型：${modelDetail.model_name}`}>
                <VolatilityOverlay
                  models={[
                    {
                      name: modelDetail.model_name,
                      volatility: modelDetail.conditional_volatility,
                    },
                  ]}
                />
              </ChartWrapper>
            </>
          )}
        </Section>

        <div className="section-divider" />

        {/* ── Section 3: SO WHAT ─────────────────────────────── */}
        <Section id="so-what" index={3} title="SO WHAT — 诊断与交叉验证">
          {modelDetail && diagnostics && (
            <>
              <ChartWrapper title="残差诊断">
                <ResidualDiagnostics
                  residuals={residuals}
                  diagnostics={diagnostics}
                />
              </ChartWrapper>
              <ReadGuide>
                <p>
                  <strong>四面板诊断：</strong>(1) 标准残差时序 — 应无趋势和异常聚集；
                  (2) QQ 图 — 点沿直线表示近似正态；
                  (3) 残差² — 检验 ARCH 效应，若显著则模型未完全捕捉波动率聚集；
                  (4) 标准化残差 — 应近似白噪声。
                </p>
              </ReadGuide>

              {modelDetail.params && (
                <div className="mt-8">
                  <h3 className="text-lg font-semibold text-foreground mb-3">
                    关键参数解读
                  </h3>
                  <div className="space-y-3">
                    {modelDetail.params.omega != null && (
                      <p>
                        <ParamTooltip
                          name="ω (常数项)"
                          value={modelDetail.params.omega}
                          tooltip="长期方差水平，值越大表示基础波动率越高。"
                        />
                      </p>
                    )}
                    {modelDetail.params.alpha != null && (
                      <p>
                        <ParamTooltip
                          name="α (ARCH 项)"
                          value={modelDetail.params.alpha}
                          tooltip="对滞后冲击的敏感度，α 越大表示波动率对近期事件反应越剧烈。"
                        />
                      </p>
                    )}
                    {modelDetail.params.beta != null && (
                      <p>
                        <ParamTooltip
                          name="β (GARCH 项)"
                          value={modelDetail.params.beta}
                          tooltip="波动率持续性，β 越大表示冲击衰减越慢，高波动期持续越久。"
                        />
                      </p>
                    )}
                    {modelDetail.persistence != null && (
                      <p>
                        <ParamTooltip
                          name="α + β (持续性)"
                          value={modelDetail.persistence}
                          tooltip={
                            <>
                              持续性参数。若接近 1（如 0.95+），冲击衰减极慢，半衰期可达数十天；
                              若 &gt; 1，方差非平稳（IGARCH 边界）。
                            </>
                          }
                        />
                      </p>
                    )}
                    {modelDetail.params.gamma != null && (
                      <p>
                        <ParamTooltip
                          name="γ (非对称项)"
                          value={modelDetail.params.gamma}
                          tooltip="杠杆效应。γ > 0 表示负冲击（利差走阔）对波动率的放大作用强于正冲击。"
                        />
                      </p>
                    )}
                  </div>
                </div>
              )}
            </>
          )}

          {/* Cross-model diagnostics */}
          <InsightCard title="前沿 vs 经典：方法对比矩阵" variant="info">
            <div className="overflow-x-auto mt-3">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border/40">
                    <th className="text-left py-1 pr-2">模型</th>
                    <th className="text-left py-1 pr-2">年份</th>
                    <th className="text-left py-1 pr-2">记忆结构</th>
                    <th className="text-left py-1 pr-2">不确定性</th>
                    <th className="text-left py-1">计算</th>
                  </tr>
                </thead>
                <tbody className="text-muted-foreground">
                  <tr className="border-b border-border/20">
                    <td className="py-1 pr-2 font-medium text-foreground">GARCH</td>
                    <td className="py-1 pr-2">1986</td>
                    <td className="py-1 pr-2">单指数衰减</td>
                    <td className="py-1 pr-2">无</td>
                    <td className="py-1">低</td>
                  </tr>
                  <tr className="border-b border-border/20">
                    <td className="py-1 pr-2 font-medium text-foreground">HAR-RV</td>
                    <td className="py-1 pr-2">2009</td>
                    <td className="py-1 pr-2">三窗口异质</td>
                    <td className="py-1 pr-2">无(OLS)</td>
                    <td className="py-1">极低</td>
                  </tr>
                  <tr className="border-b border-border/20">
                    <td className="py-1 pr-2 font-medium text-foreground">SV</td>
                    <td className="py-1 pr-2">1986/98</td>
                    <td className="py-1 pr-2">潜在 AR(1)</td>
                    <td className="py-1 pr-2">有(后验)</td>
                    <td className="py-1">高</td>
                  </tr>
                  <tr className="border-b border-border/20">
                    <td className="py-1 pr-2 font-medium text-foreground">GAS</td>
                    <td className="py-1 pr-2">2013</td>
                    <td className="py-1 pr-2">Score-driven</td>
                    <td className="py-1 pr-2">有(Fisher)</td>
                    <td className="py-1">低</td>
                  </tr>
                  <tr>
                    <td className="py-1 pr-2 font-medium text-foreground">MS-GARCH</td>
                    <td className="py-1 pr-2">2012</td>
                    <td className="py-1 pr-2">状态切换</td>
                    <td className="py-1 pr-2">有(状态概率)</td>
                    <td className="py-1">中</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </InsightCard>
        </Section>

        <div className="section-divider" />

        {/* ── Section 4: NOW WHAT ────────────────────────────── */}
        <Section id="now-what" index={4} title="推荐模型与投资含义">
          {tournament?.winner_aic && (
            <InsightCard
              title={`AIC 最优：${tournament.winner_aic}`}
              variant="success"
            >
              <p>
                基于 AIC 准则，<strong>{tournament.winner_aic}</strong>{" "}
                在拟合优度与复杂度之间取得最佳平衡。
                建议将其作为后续风险度量（VaR / ES）和情景分析的基础模型。
              </p>
            </InsightCard>
          )}

          <ProseBlock>
            <p>
              GARCH 族 + 前沿模型的条件波动率估计为风险度量提供时变输入。
              相比历史标准差的常数假设，条件 VaR 可在高波动期自动放大风险预算，
              在低波动期收紧，实现动态风险管理。
            </p>
            <p className="mt-4">投资含义：</p>
            <ul className="list-disc list-inside mt-2 space-y-1">
              <li>
                <strong>波动率择时：</strong>条件波动率高企时减持长久期城投债，低波动期增持；
              </li>
              <li>
                <strong>风险预算：</strong>使用 SV 后验上界作为保守 VaR 输入，
                避免固定比例止损在波动率跳升时过早触发；
              </li>
              <li>
                <strong>压力测试：</strong>MS-GARCH 状态切换概率可用于构建
                前瞻性压力场景——从当前状态出发，模拟转移到高波状态的概率和路径；
              </li>
              <li>
                <strong>模型集成：</strong>HAR-RV 的三窗口结构可用于实时波动率监控，
                当 <Formula math="\beta_m \cdot RV^{(m)}" /> 显著上升时预警系统性风险。
              </li>
            </ul>
          </ProseBlock>

          <PageNavigation
            prev={{ href: "/analysis/overview", label: "概览", emoji: "📊" }}
            next={{ href: "/analysis/risk", label: "风险度量", emoji: "⚠️" }}
          />
        </Section>
      </div>

      {/* Right-side TOC */}
      <aside className="hidden xl:block w-56 shrink-0 border-l border-border/30 py-8 pr-4">
        <div className="sticky top-20 space-y-1">
          <p className="text-[10px] uppercase tracking-widest text-muted-foreground/70 mb-3 font-semibold">
            目录
          </p>
          {TOC_SECTIONS.map((s) => (
            <a
              key={s.id}
              href={`#${s.id}`}
              className={`
                block px-3 py-1.5 rounded-md text-xs transition-all duration-200
                ${
                  activeSection === s.id
                    ? "bg-primary/10 text-primary font-medium border-l-2 border-primary pl-2"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                }
              `}
            >
              <span className="font-mono text-[10px] opacity-60 mr-1">
                {String(TOC_SECTIONS.indexOf(s)).padStart(2, "0")}
              </span>
              {s.label}
              <span className="block text-[10px] opacity-50 ml-4">{s.sublabel}</span>
            </a>
          ))}
        </div>
      </aside>

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
