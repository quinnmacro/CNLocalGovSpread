"use client";

import { MetricCard } from "@/components/narrative/metric-card";
import { useMarketGauge, useRiskMetrics, useDataSummary } from "@/hooks/use-api";
import { Skeleton } from "@/components/ui/skeleton";
import { Activity, Percent, Gauge, ShieldAlert } from "lucide-react";

export function LiveSnapshot() {
  const gauge = useMarketGauge();
  const risk = useRiskMetrics(0.99);
  const summary = useDataSummary();

  // Extract latest spread value from data summary (spread_all median as proxy)
  const latestSpread =
    summary.data?.summary_stats?.spread_all?.median ?? null;

  // Compute approximate percentile (using mean/std as a rough guide)
  const mean = summary.data?.summary_stats?.spread_all?.mean;
  const std = summary.data?.summary_stats?.spread_all?.std;

  // Gauge status: [english, chinese]
  const statusEn = gauge.data?.status?.[0] ?? "Loading";
  const statusZh = gauge.data?.status?.[1] ?? "加载中";
  const composite = gauge.data?.composite;

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {/* Current spread */}
      {summary.isLoading ? (
        <Skeleton className="h-28 rounded-lg" />
      ) : (
        <MetricCard
          label="当前利差"
          value={latestSpread}
          unit="bps"
          decimals={2}
          icon={<Activity className="h-4 w-4" />}
          changeLabel="median"
        />
      )}

      {/* Historical percentile */}
      {summary.isLoading ? (
        <Skeleton className="h-28 rounded-lg" />
      ) : (
        <MetricCard
          label="历史分位"
          value={
            mean != null && std != null && latestSpread != null
              ? `${(() => {
                const n = Math.round(((latestSpread - mean) / std) * 20 + 50);
                const s = ["th", "st", "nd", "rd"];
                const v = n % 100;
                return n + (s[(v - 20) % 10] || s[v] || s[0]);
              })()}`
              : "—"
          }
          decimals={0}
          icon={<Percent className="h-4 w-4" />}
          changeLabel="percentile"
        />
      )}

      {/* Market state */}
      {gauge.isLoading ? (
        <Skeleton className="h-28 rounded-lg" />
      ) : (
        <MetricCard
          label="市场状态"
          value={statusZh}
          decimals={0}
          icon={<Gauge className="h-4 w-4" />}
          changeLabel="composite"
        />
      )}

      {/* VaR 99% */}
      {risk.isLoading ? (
        <Skeleton className="h-28 rounded-lg" />
      ) : (
        <MetricCard
          label="VaR 99%"
          value={risk.data?.var_evt ?? null}
          unit="bps"
          decimals={3}
          icon={<ShieldAlert className="h-4 w-4" />}
          changeLabel="EVT"
        />
      )}
    </div>
  );
}
