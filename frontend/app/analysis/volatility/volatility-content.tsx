"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/layout/sidebar";
import { Breadcrumb } from "@/components/layout/breadcrumb";
import { Section } from "@/components/narrative/section";
import { ProseBlock } from "@/components/narrative/prose-block";
import { ReadGuide } from "@/components/narrative/read-guide";
import { InsightCard } from "@/components/narrative/insight-card";
import { ParamTooltip } from "@/components/narrative/param-tooltip";
import { Formula } from "@/components/narrative/formula";
import { VolatilityOverlay } from "@/components/charts/volatility-overlay";
import { TournamentTable } from "@/components/charts/tournament-table";
import { ResidualDiagnostics } from "@/components/charts/residual-diagnostics";
import { ModelSelector } from "@/components/interactive/model-selector";
import { Skeleton } from "@/components/ui/skeleton";
import {
  useTournament,
  useModelDetail,
  useFigarch,
  useFitCustom,
} from "@/hooks/use-api";
import { fmt } from "@/lib/utils";
import type { TimePoint, CustomFitRequest } from "@/lib/types";
import Link from "next/link";
import { ArrowRight } from "lucide-react";

const PAGE_INFO = {
  title: "波动率建模",
  subtitle: "Volatility — GARCH 族模型对比与诊断",
};

export function VolatilityContent() {
  const { data: tournament, isLoading: tournamentLoading, error: tournamentError } = useTournament();
  const { data: figarch, isLoading: figarchLoading, error: figarchError } = useFigarch();
  const fitCustom = useFitCustom();

  // Default to winner_aic or first model
  const defaultModel = tournament?.winner_aic?.toLowerCase() ?? tournament?.models[0]?.model_name?.toLowerCase() ?? "garch";
  const [selectedModel, setSelectedModel] = useState<string>(defaultModel);

  // Update selected model when tournament data loads
  useEffect(() => {
    if (tournament?.winner_aic && selectedModel === defaultModel) {
      setSelectedModel(tournament.winner_aic);
    }
  }, [tournament, selectedModel, defaultModel]);

  const { data: modelDetail, isLoading: detailLoading } = useModelDetail(selectedModel?.toLowerCase(), {
    enabled: !!selectedModel && /^(garch|egarch|gjr|ewma|figarch)$/.test(selectedModel.toLowerCase()),
  });

  const isLoading = tournamentLoading || figarchLoading;
  const hasError = tournamentError || figarchError;

  // Collect volatility series from all models
  const volatilityModels: { name: string; volatility: TimePoint[] }[] = [];
  if (tournament?.models) {
    tournament.models.forEach((m) => {
      // We'll fetch details for all models in parallel using multiple hooks
      // For now, just use the selected model and figarch
    });
  }
  if (modelDetail?.conditional_volatility) {
    volatilityModels.push({ name: modelDetail.model_name, volatility: modelDetail.conditional_volatility });
  }
  if (figarch?.conditional_volatility) {
    volatilityModels.push({ name: figarch.model_name, volatility: figarch.conditional_volatility });
  }

  // Get standardized residuals for diagnostics
  const residuals = modelDetail?.standardized_residuals ?? [];
  const diagnostics = modelDetail?.diagnostics ?? null;

  function handleFitCustom(request: CustomFitRequest) {
    fitCustom.mutate(request, {
      onSuccess: (result) => {
        setSelectedModel(result.model_name);
      },
    });
  }

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
        <Section index={0} title="为何需要条件波动率模型？">
          <ProseBlock>
            <p>
              金融时间序列普遍存在 <strong>波动率聚集（Volatility Clustering）</strong> 现象：
              高波动时期后跟随高波动，低波动时期后跟随低波动。传统的历史标准差假设波动率为常数，
              无法捕捉这种时变特征，导致风险低估或高估。
            </p>
            <p>
              城投债利差同样表现出显著的波动率聚集：2020 年疫情冲击期间利差波动率飙升至正常水平的
              3-5 倍，随后逐步回归。条件波动率模型（Conditional Volatility Models）通过引入
              滞后项动态刻画这种时变行为，为风险度量和情景分析提供更准确的输入。
            </p>
          </ProseBlock>
        </Section>

        {/* Section 1: HOW — 方法论 */}
        <Section index={1} title="GARCH 族模型" subtitle="Model Specifications">
          <ProseBlock>
            <p>
              本模块实现 4 类 GARCH 模型，覆盖对称与非对称效应、短期与长期记忆。
              所有模型通过极大似然估计（MLE）拟合，残差分布支持正态、t 分布和偏 t 分布。
            </p>
          </ProseBlock>

          <h3 className="text-lg font-semibold text-foreground mb-3 mt-8">
            GARCH(1,1) — 基准模型
          </h3>
          <Formula
            block
            math={String.raw`\sigma^2_t = \omega + \alpha \epsilon^2_{t-1} + \beta \sigma^2_{t-1}`}
            className="my-4"
          />
          <ProseBlock>
            <p>
              最经典的 GARCH 模型。条件方差由常数项 <Formula math={String.raw`\omega`} />、
              滞后残差平方 <Formula math={String.raw`\alpha \epsilon^2_{t-1}`} />（ARCH 项）和
              滞后方差 <Formula math={String.raw`\beta \sigma^2_{t-1}`} />（GARCH 项）组成。
              持续性由 <Formula math={String.raw`\alpha + \beta`} /> 衡量。
            </p>
          </ProseBlock>

          <h3 className="text-lg font-semibold text-foreground mb-3 mt-8">
            EGARCH(1,1) — 指数 GARCH
          </h3>
          <Formula
            block
            math={String.raw`\ln(\sigma^2_t) = \omega + \alpha \left(\frac{|\epsilon_{t-1}|}{\sigma_{t-1}} - \sqrt{\frac{2}{\pi}}\right) + \gamma \frac{\epsilon_{t-1}}{\sigma_{t-1}} + \beta \ln(\sigma^2_{t-1})`}
            className="my-4"
          />
          <ProseBlock>
            <p>
              Nelson (1991) 提出。对数形式保证方差非负，无需参数约束。非对称项
              <Formula math={String.raw`\gamma`} /> 捕捉杠杆效应：负冲击（
              <Formula math={String.raw`\epsilon_{t-1} < 0`} />）对波动率的影响大于正冲击。
            </p>
          </ProseBlock>

          <h3 className="text-lg font-semibold text-foreground mb-3 mt-8">
            GJR-GARCH(1,1) — 非对称 GARCH
          </h3>
          <Formula
            block
            math={String.raw`\sigma^2_t = \omega + (\alpha + \gamma I_{t-1}) \epsilon^2_{t-1} + \beta \sigma^2_{t-1}`}
            className="my-4"
          />
          <ProseBlock>
            <p>
              Glosten, Jagannathan, Runkle (1993) 提出。通过指示函数
              <Formula math={String.raw`I_{t-1} = \mathbb{1}(\epsilon_{t-1} < 0)`} /> 引入非对称效应。
              当 <Formula math={String.raw`\gamma > 0`} /> 时，负冲击放大波动率。
            </p>
          </ProseBlock>

          <h3 className="text-lg font-semibold text-foreground mb-3 mt-8">
            FIGARCH — 分数积分 GARCH
          </h3>
          <Formula
            block
            math={String.raw`(1-\beta L)\sigma^2_t = \omega + (1-\beta L - (1-\phi L)^d)\epsilon^2_t`}
            className="my-4"
          />
          <ProseBlock>
            <p>
              Baillie, Bollerslev, Mikkelsen (1996) 提出。分数差分参数
              <Formula math={String.raw`d \in (0,1)`} /> 捕捉长期记忆（Long Memory）：
              波动率冲击以双曲速率衰减，而非标准 GARCH 的指数速率。适用于持续性极强的利差序列。
            </p>
          </ProseBlock>

          <InsightCard title="模型选择准则" variant="info">
            <p>
              使用 <strong>AIC（Akaike Information Criterion）</strong> 和{" "}
              <strong>BIC（Bayesian Information Criterion）</strong> 权衡拟合优度与模型复杂度。
              两者均惩罚参数数量，但 BIC 惩罚更重，倾向选择更简洁的模型。
              最优模型 = AIC / BIC 最小者。
            </p>
          </InsightCard>
        </Section>

        {/* Section 2: WHAT — 结果展示 */}
        <Section index={2} title="模型对比与条件波动率" subtitle="Tournament & Volatility Overlay">
          {tournament && (
            <>
              <TournamentTable
                models={tournament.models}
                winnerAic={tournament.winner_aic}
                winnerBic={tournament.winner_bic}
                onModelClick={setSelectedModel}
                selectedModel={selectedModel}
              />
              <ReadGuide>
                <p>
                  <strong>表格解读：</strong>AIC / BIC 越小越好；持续性 = α + β，越接近 1 表示波动率冲击衰减越慢；
                  ARCH p &lt; 0.05 表示残差仍存在 ARCH 效应（模型未完全捕捉波动率聚集）；
                  JB p &lt; 0.05 表示残差非正态。
                </p>
              </ReadGuide>
            </>
          )}

          <h3 className="text-lg font-semibold text-foreground mb-3 mt-12">
            条件波动率叠加图
          </h3>
          <VolatilityOverlay models={volatilityModels} height={500} />
          <ReadGuide>
            <p>
              <strong>读图要点：</strong>实线为各模型的条件波动率估计。注意波动率聚集特征：
              2020 年初和 2022 年底出现两次显著跳升。不同模型的波动率估计在平稳期接近，
              但在极端时期可能分歧，反映模型对尾部事件的不同响应。
            </p>
          </ReadGuide>

          <h3 className="text-lg font-semibold text-foreground mb-3 mt-8">
            自定义模型拟合
          </h3>
          <ModelSelector onFit={handleFitCustom} isFitting={fitCustom.isPending} />
          {fitCustom.data && (
            <InsightCard
              title={`自定义拟合结果：${fitCustom.data.model_name}`}
              variant="success"
            >
              <p>
                AIC = {fmt(fitCustom.data.aic, 2)}，BIC = {fmt(fitCustom.data.bic, 2)}，
                {fitCustom.data.converged ? "已收敛" : "未收敛"}。
              </p>
            </InsightCard>
          )}
        </Section>

        {/* Section 3: SO WHAT — 诊断与解读 */}
        <Section index={3} title="残差诊断与参数解读" subtitle="Residual Diagnostics">
          {detailLoading ? (
            <Skeleton className="h-96 w-full" />
          ) : modelDetail && residuals.length > 0 ? (
            <>
              <h3 className="text-lg font-semibold text-foreground mb-3">
                四面板诊断图
              </h3>
              <ResidualDiagnostics
                residuals={residuals}
                diagnostics={diagnostics}
                height={700}
              />
              <ReadGuide>
                <p>
                  <strong>四面板解读：</strong>
                  (1) QQ Plot — 点偏离对角线表示残差非正态；
                  (2) ACF 残差 — 自相关函数应在置信带内，否则存在序列相关；
                  (3) ACF 残差² — 检验 ARCH 效应，若显著则模型未完全捕捉波动率聚集；
                  (4) 标准化残差 — 应近似白噪声，无趋势或聚集。
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
          ) : (
            <p className="text-muted-foreground">请选择模型以查看诊断结果。</p>
          )}
        </Section>

        {/* Section 4: NOW WHAT — 投资含义 */}
        <Section index={4} title="推荐模型与投资含义">
          {tournament?.winner_aic && (
            <InsightCard title={`AIC 最优：${tournament.winner_aic}`} variant="success">
              <p>
                基于 AIC 准则，<strong>{tournament.winner_aic}</strong> 在拟合优度与复杂度之间取得最佳平衡。
                建议将其作为后续风险度量（VaR / ES）和情景分析的基础模型。
              </p>
            </InsightCard>
          )}

          <ProseBlock>
            <p>
              GARCH 族模型的条件波动率估计为风险度量提供时变输入。相比历史标准差的常数假设，
              条件 VaR 可在高波动期自动放大风险预算，在低波动期收紧，实现动态风险管理。
            </p>
            <p className="mt-4">
              投资含义：
            </p>
            <ul className="list-disc list-inside mt-2 space-y-1">
              <li>
                <strong>波动率择时：</strong>条件波动率高企时减持长久期城投债，低波动期增持；
              </li>
              <li>
                <strong>风险预算：</strong>使用条件 VaR 动态调整头寸规模，避免固定比例止损在波动率跳升时过早触发；
              </li>
              <li>
                <strong>压力测试：</strong>将条件波动率作为蒙特卡洛模拟的输入，生成时变情景分布。
              </li>
            </ul>
          </ProseBlock>

          <Link
            href="/analysis/risk"
            className="inline-flex items-center gap-2 mt-6 px-4 py-2 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            进入风险度量
            <ArrowRight className="h-4 w-4" />
          </Link>
        </Section>
      </div>
    </div>
  );
}
