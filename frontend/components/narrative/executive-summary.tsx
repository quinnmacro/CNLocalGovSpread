"use client";

import { cn, fmt } from "@/lib/utils";
import { motion } from "framer-motion";
import {
  TrendingUp,
  TrendingDown,
  Shield,
  Activity,
  AlertTriangle,
  CheckCircle,
  Target,
  Layers,
  Brain,
} from "lucide-react";

interface ExecutiveSummaryProps {
  kalmanZ: number;
  stsZ: number;
  bayesZ: number;
  currentRegime: string;
  gaugeComposite: number;
  kalmanOvervalued: boolean;
  stsOvervalued: boolean;
  bayesOvervalued: boolean;
}

type Verdict = "overvalued" | "undervalued" | "neutral";

function computeVerdict(
  props: ExecutiveSummaryProps,
): { verdict: Verdict; agreement: number; label: string } {
  const overCount = [
    props.kalmanOvervalued,
    props.stsOvervalued,
    props.bayesOvervalued,
  ].filter(Boolean).length;
  const underCount = [
    props.kalmanZ < -1.5,
    props.stsZ < -1.5,
    props.bayesZ < -1.5,
  ].filter(Boolean).length;

  if (overCount >= 2) {
    return {
      verdict: "overvalued",
      agreement: overCount,
      label: `${overCount}/3 方法判断利差高估`,
    };
  }
  if (underCount >= 2) {
    return {
      verdict: "undervalued",
      agreement: underCount,
      label: `${underCount}/3 方法判断利差低估`,
    };
  }
  return { verdict: "neutral", agreement: 3, label: "多数方法认为利差处于合理区间" };
}

function verdictBadge(v: Verdict) {
  switch (v) {
    case "overvalued":
      return { cls: "exec-verdict exec-verdict-overvalued", icon: TrendingUp, text: "利差高估" };
    case "undervalued":
      return { cls: "exec-verdict exec-verdict-undervalued", icon: TrendingDown, text: "利差低估" };
    case "neutral":
      return { cls: "exec-verdict exec-verdict-neutral", icon: Shield, text: "利差合理" };
  }
}

function zColor(z: number): string {
  if (z > 1.5) return "text-chart-3";
  if (z < -1.5) return "text-chart-2";
  return "text-foreground";
}

function regimeIcon(regime: string) {
  if (regime.includes("高")) return AlertTriangle;
  if (regime.includes("中")) return Activity;
  return CheckCircle;
}

function regimeColor(regime: string): string {
  if (regime.includes("高")) return "text-chart-3";
  if (regime.includes("中")) return "text-chart-3";
  return "text-chart-2";
}

function gaugeColor(score: number): string {
  if (score >= 60) return "text-destructive";
  if (score >= 30) return "text-chart-3";
  return "text-chart-2";
}

function gaugeLabel(score: number): string {
  if (score >= 80) return "Crisis";
  if (score >= 60) return "Stress";
  if (score >= 30) return "Caution";
  return "Calm";
}

export function ExecutiveSummary(props: ExecutiveSummaryProps) {
  const { verdict, agreement, label } = computeVerdict(props);
  const badge = verdictBadge(verdict);
  const BadgeIcon = badge.icon;
  const RegimeIcon = regimeIcon(props.currentRegime);

  const agreementPct = (agreement / 3) * 100;
  const agreementColor =
    verdict === "overvalued"
      ? "bg-chart-3"
      : verdict === "undervalued"
        ? "bg-chart-2"
        : "bg-primary";

  const summaryText = (() => {
    switch (verdict) {
      case "overvalued":
        return "多方法交叉验证显示当前利差显著低于基本面水平，存在均值回归（走阔）的交易机会，但需关注短期流动性和政策风险。";
      case "undervalued":
        return "多方法交叉验证显示当前利差显著高于基本面水平，收窄趋势可能持续，适合增加城投债配置。";
      case "neutral":
        return "各方法信号一致，当前利差与基本面水平基本吻合，建议维持现有配置，关注后续信号变化。";
    }
  })();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className="exec-summary mb-8"
    >
      <div className="exec-summary-inner">
        {/* Top row: Verdict badge */}
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-3">
            <span className={cn("text-xs font-mono text-muted-foreground uppercase tracking-wider")}>
              综合诊断
            </span>
            <span className={badge.cls}>
              <BadgeIcon className="h-4 w-4" />
              {badge.text}
            </span>
          </div>
          <span className="text-[10px] text-muted-foreground/60 font-mono hidden sm:inline">
            基于 Kalman · STS · Bayesian · HMM · MarketGauge
          </span>
        </div>

        {/* Mini signal cards */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2.5 mb-5">
          <div className="exec-mini-card">
            <div className="exec-mini-label">Kalman z</div>
            <div className={cn("exec-mini-value", zColor(props.kalmanZ))}>
              {props.kalmanZ >= 0 ? "+" : ""}
              {fmt(props.kalmanZ, 3)}
            </div>
            <div className="exec-mini-detail">
              {Math.abs(props.kalmanZ) > 1.5 ? "⚠ 极端" : "✓ 正常"}
            </div>
          </div>
          <div className="exec-mini-card">
            <div className="exec-mini-label">STS z</div>
            <div className={cn("exec-mini-value", zColor(props.stsZ))}>
              {props.stsZ >= 0 ? "+" : ""}
              {fmt(props.stsZ, 3)}
            </div>
            <div className="exec-mini-detail">
              {Math.abs(props.stsZ) > 1.5 ? "⚠ 极端" : "✓ 正常"}
            </div>
          </div>
          <div className="exec-mini-card">
            <div className="exec-mini-label">Bayesian z</div>
            <div className={cn("exec-mini-value", zColor(props.bayesZ))}>
              {props.bayesZ >= 0 ? "+" : ""}
              {fmt(props.bayesZ, 3)}
            </div>
            <div className="exec-mini-detail">
              {Math.abs(props.bayesZ) > 1.5 ? "⚠ 极端" : "✓ 正常"}
            </div>
          </div>
          <div className="exec-mini-card">
            <div className="exec-mini-label">HMM 状态</div>
            <div className={cn("exec-mini-value flex items-center gap-1.5", regimeColor(props.currentRegime))}>
              <RegimeIcon className="h-4 w-4" />
              <span className="text-sm">{props.currentRegime}</span>
            </div>
            <div className="exec-mini-detail">三状态隐马尔可夫</div>
          </div>
          <div className="exec-mini-card">
            <div className="exec-mini-label">MarketGauge</div>
            <div className={cn("exec-mini-value", gaugeColor(props.gaugeComposite))}>
              {fmt(props.gaugeComposite, 1)}
            </div>
            <div className="exec-mini-detail">{gaugeLabel(props.gaugeComposite)}</div>
          </div>
        </div>

        {/* Agreement bar */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-xs text-muted-foreground flex items-center gap-1.5">
              <Layers className="h-3 w-3" />
              信号一致性
            </span>
            <span className="text-xs font-mono text-foreground">{label}</span>
          </div>
          <div className="exec-agreement-bar">
            <motion.div
              className={cn("exec-agreement-fill", agreementColor)}
              initial={{ width: 0 }}
              animate={{ width: `${agreementPct}%` }}
              transition={{ duration: 0.8, ease: "easeOut", delay: 0.3 }}
            />
          </div>
        </div>

        {/* Summary text */}
        <div className="flex items-start gap-2 pt-3 border-t border-border/30">
          <Target className="h-4 w-4 text-primary/60 mt-0.5 shrink-0" />
          <p className="text-sm text-muted-foreground leading-relaxed">
            {summaryText}
          </p>
        </div>
      </div>
    </motion.div>
  );
}
