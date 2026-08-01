import Link from "next/link";
import {

  BarChart3,
  LineChart,
  Shield,
  Repeat,
  Sparkles,
  ArrowRight,
  Database,
  Calculator,
  AlertTriangle,
  GitBranch,
  Target,
} from "lucide-react";
import { NavigationCard } from "@/components/narrative/navigation-card";
import { LiveSnapshot } from "./_components/live-snapshot";
import { HeroDataBadge } from "./_components/hero-data-badge";
import { Separator } from "@/components/ui/separator";

const modules = [
  {
    href: "/analysis/overview",
    emoji: "📊",
    title: "利差全景",
    titleEn: "Overview",
    description:
      "地方债信用利差的趋势、分布、期限结构全景分析。厚尾、尖峰、状态切换——数据讲了什么故事？",
  },
  {
    href: "/analysis/volatility",
    emoji: "📈",
    title: "波动率建模",
    titleEn: "Volatility",
    description:
      "7 种条件异方差模型大对比：GARCH / EGARCH / GJR / FIGARCH(FFT) / EWMA / Kalman / ML。哪个模型最能描述利差的波动？",
  },
  {
    href: "/analysis/risk",
    emoji: "⚠️",
    title: "风险度量",
    titleEn: "Risk",
    description:
      "VaR / Expected Shortfall / 极值理论 (GPD-POT)。尾部有多厚？Hill 估计告诉你。",
  },
  {
    href: "/analysis/regimes",
    emoji: "🔄",
    title: "市场状态",
    titleEn: "Regimes",
    description:
      "HMM + STS + Bayesian STS + CPD 多维状态识别。趋势信号、偏离度、后验不确定性与结构性断裂点——当前市场处于什么状态？",
  },
  {
    href: "/analysis/scenarios",
    emoji: "🔮",
    title: "情景分析",
    titleEn: "Scenarios",
    description:
      "AR(1)+GARCH 蒙特卡洛模拟：未来 1 年利差可能怎么走？压力测试下的持仓表现如何？",
  },
];

export default function HomePage() {
  return (
    <div className="relative">
      {/* ─── Hero ──────────────────────────────────────────── */}
      <section className="relative overflow-hidden border-b border-border/40">
        {/* Background gradient mesh */}
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-chart-2/5 pointer-events-none" />
        <div className="absolute top-20 -left-20 w-96 h-96 bg-primary/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 right-0 w-80 h-80 bg-chart-4/5 rounded-full blur-3xl pointer-events-none" />

        <div className="relative max-w-5xl mx-auto px-4 md:px-6 py-20 md:py-28">
          <div className="flex flex-col items-center text-center">
            <div className="inline-flex items-center gap-2 bg-primary/10 border border-primary/20 rounded-full px-4 py-1.5 mb-6">
              <HeroDataBadge />
            </div>

            <h1 className="text-4xl md:text-6xl font-bold tracking-tight text-foreground mb-4">
              中国地方政府债
              <br />
              <span className="bg-gradient-to-r from-primary to-chart-2 bg-clip-text text-transparent">
                信用利差
              </span>
            </h1>

            <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mb-8 leading-relaxed">
              基于{" "}
              <span className="font-mono text-primary/80">GARCH</span> /{" "}
              <span className="font-mono text-primary/80">EVT</span> /{" "}
              <span className="font-mono text-primary/80">HMM</span> /{" "}
              <span className="font-mono text-primary/80">STS</span> /{" "}
              <span className="font-mono text-primary/80">Bayesian</span>{" "}
              的利差建模、风险度量与状态监控系统
            </p>

            <div className="flex flex-col sm:flex-row gap-3">
              <Link
                href="/analysis/overview"
                className="inline-flex items-center justify-center gap-2 px-6 py-3 rounded-lg bg-primary text-primary-foreground font-medium hover:bg-primary/90 transition-colors"
              >
                进入分析
                <ArrowRight className="h-4 w-4" />
              </Link>
              <a
                href="https://quinnmacro.com"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center gap-2 px-6 py-3 rounded-lg border border-border text-foreground hover:bg-muted transition-colors"
              >
                About QuinnMacro
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* ─── Live Snapshot ────────────────────────────────── */}
      <section className="border-b border-border/40 bg-card/30">
        <div className="max-w-5xl mx-auto px-4 md:px-6 py-12">
          <div className="flex items-center gap-2 mb-6">
            <Sparkles className="h-4 w-4 text-primary" />
            <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
              实时市场快照 · Live Snapshot
            </h2>
          </div>
          <LiveSnapshot />
        </div>
      </section>

      {/* ─── Research Abstract ────────────────────────────── */}
      <section className="border-b border-border/40">
        <div className="max-w-4xl mx-auto px-4 md:px-6 py-16">
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight text-foreground mb-6">
            研究摘要
          </h2>
          <div className="prose-narrative space-y-4 text-muted-foreground">
            <p>
              中国地方政府债是固收市场最大的品种之一（存量规模超 40 万亿元人民币）。
              信用利差（地方债收益率 − 同期限国债收益率）是市场对地方政府信用风险的核心定价变量。
              理解利差的波动特征、尾部风险和市场状态，对组合管理、风险控制和策略研究具有直接意义。
            </p>
            <p>
              本平台基于日频利差数据，构建了完整的量化分析流水线：
            </p>
            <ul className="space-y-2 pl-4 border-l-2 border-primary/30 ml-2">
              <li className="flex items-start gap-2">
                <span className="text-primary font-mono text-sm mt-0.5">01</span>
                <span>
                  <strong className="text-foreground">波动率建模</strong> — 7 种条件异方差模型对比
                  (含 FFT 优化的 FIGARCH、Kalman 滤波、机器学习)
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary font-mono text-sm mt-0.5">02</span>
                <span>
                  <strong className="text-foreground">尾部风险</strong> — GPD-POT 极值理论 + Hill
                  估计 + VaR/ES 回测 (Kupiec + Christoffersen)
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary font-mono text-sm mt-0.5">03</span>
                <span>
                  <strong className="text-foreground">状态识别</strong> — HMM + 结构化时间序列 (STS)
                  + 贝叶斯推断 + 结构性变化点检测
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-primary font-mono text-sm mt-0.5">04</span>
                <span>
                  <strong className="text-foreground">前瞻分析</strong> — AR(1)+GARCH 蒙特卡洛
                  情景生成 + 波动率冲击压力测试
                </span>
              </li>
            </ul>
            <p>
              引擎基于 Python 科学计算栈，53 个单元/集成测试全过，代码完全开源。
              本页面的每一张图表都附带<strong className="text-foreground">读图指南</strong>
              和经济学注解——我们希望这不只是一个展示数据的 dashboard，而是一份有叙事的研究笔记。
            </p>
          </div>
        </div>
      </section>

      {/* ─── Framework Overview ───────────────────────────── */}
      <section className="border-b border-border/40 bg-card/20">
        <div className="max-w-5xl mx-auto px-4 md:px-6 py-16">
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight text-foreground mb-8 text-center">
            框架架构
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4 items-stretch">
            <FrameworkStep
              icon={<Database className="h-5 w-5" />}
              title="Raw Data"
              subtitle="Wind EDB"
              detail="日频利差 4 品种"
            />
            <FrameworkStep
              icon={<Calculator className="h-5 w-5" />}
              title="Volatility"
              subtitle="7 Models"
              detail="GARCH · EGARCH · FIGARCH"
            />
            <FrameworkStep
              icon={<AlertTriangle className="h-5 w-5" />}
              title="Risk"
              subtitle="VaR / ES / EVT"
              detail="Hill · GPD-POT · Backtest"
            />
            <FrameworkStep
              icon={<GitBranch className="h-5 w-5" />}
              title="Regimes"
              subtitle="HMM+STS+Bayes+CPD"
              detail="多维状态 · 信号提取"
            />
            <FrameworkStep
              icon={<Target className="h-5 w-5" />}
              title="Scenarios"
              subtitle="Monte Carlo"
              detail="压力测试 · 扇形图"
            />
          </div>
        </div>
      </section>

      {/* ─── Navigation Cards ─────────────────────────────── */}
      <section className="max-w-5xl mx-auto px-4 md:px-6 py-16">
        <div className="text-center mb-10">
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight text-foreground mb-2">
            探索分析模块
          </h2>
          <p className="text-muted-foreground">
            每个模块遵循{" "}
            <span className="font-mono text-primary/80">
              WHY → HOW → WHAT → SO WHAT → NOW WHAT
            </span>{" "}
            叙事结构
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {modules.map((m) => (
            <NavigationCard key={m.href} {...m} />
          ))}
        </div>
      </section>
    </div>
  );
}

/* ───────────────────────────────────────────────────────────── */

function FrameworkStep({
  icon,
  title,
  subtitle,
  detail,
}: {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  detail: string;
}) {
  return (
    <div className="relative group">
      <div className="bg-card border border-border/50 rounded-xl p-5 h-full flex flex-col items-center text-center gap-2 transition-all group-hover:border-primary/40 group-hover:shadow-lg group-hover:shadow-primary/5">
        <div className="text-primary mb-1">{icon}</div>
        <div className="font-bold text-foreground">{title}</div>
        <div className="text-xs font-mono text-primary/70">{subtitle}</div>
        <Separator className="my-1" />
        <div className="text-xs text-muted-foreground">{detail}</div>
      </div>
    </div>
  );
}
