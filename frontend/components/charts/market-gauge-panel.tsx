"use client";
import type { MarketGaugeResponse } from "@/lib/types";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { fmt } from "@/lib/utils";
import { cn } from "@/lib/utils";
import { Activity, AlertTriangle, CheckCircle, Info } from "lucide-react";

interface MarketGaugePanelProps {
  gauge: MarketGaugeResponse;
}

const statusConfig: Record<string, { icon: typeof Activity; color: string; bgColor: string }> = {
  calm: { icon: CheckCircle, color: "text-chart-2", bgColor: "bg-chart-2/10" },
  caution: { icon: Info, color: "text-chart-3", bgColor: "bg-chart-3/10" },
  stress: { icon: AlertTriangle, color: "text-chart-3", bgColor: "bg-chart-3/10" },
  crisis: { icon: AlertTriangle, color: "text-destructive", bgColor: "bg-destructive/10" },
};

const indicatorLabels: Record<string, string> = {
  spread_level: "利差水平",
  spread_momentum: "利差动量",
  volatility: "波动率",
  trend: "趋势强度",
  regime: "状态评分",
};

export function MarketGaugePanel({ gauge }: MarketGaugePanelProps) {
  const statusKey = gauge.status[0] ?? "caution";
  const config = statusConfig[statusKey] ?? statusConfig.caution;
  const StatusIcon = config.icon;

  return (
    <div className="space-y-6">
      {/* Main gauge display */}
      <Card className="p-6 text-center">
        <div className="flex items-center justify-center gap-3 mb-4">
          <StatusIcon className={cn("h-6 w-6", config.color)} />
          <Badge variant="outline" className={cn("text-sm px-3 py-1", config.color, "border-current/30")}>
            {gauge.status[1]}
          </Badge>
        </div>
        <div className="text-5xl font-bold font-mono text-foreground mb-2">
          {fmt(gauge.composite, 1)}
        </div>
        <div className="text-sm text-muted-foreground">综合评分 (0–100)</div>
        <Progress value={gauge.composite} className="mt-4 h-2" />
      </Card>

      {/* Sub-indicators */}
      <div className="space-y-3">
        {Object.entries(gauge.indicators).map(([key, indicator]) => (
          <Card key={key} className="p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-foreground">
                {indicatorLabels[key] ?? key}
              </span>
              <span className="text-sm font-mono text-muted-foreground">
                {fmt(indicator.score, 1)}
              </span>
            </div>
            <Progress
              value={Math.min(100, Math.max(0, indicator.score))}
              className="h-1.5"
            />
          </Card>
        ))}
      </div>
    </div>
  );
}
