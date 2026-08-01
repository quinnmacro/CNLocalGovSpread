"use client";

import { useDataSummary } from "@/hooks/use-api";
import { Activity } from "lucide-react";

export function HeroDataBadge() {
  const { data: summary } = useDataSummary();
  const nRows = summary?.n_rows ?? 2054;
  const dateStart = summary?.date_range?.[0]?.slice(0, 4) ?? "2018";
  const dateEnd = summary?.date_range?.[1]?.slice(0, 4) ?? "2023";

  return (
    <>
      <Activity className="h-3.5 w-3.5 text-primary" />
      <span className="text-xs font-medium text-primary">
        实时量化分析 · {dateStart}–{dateEnd} · {nRows} 交易日
      </span>
    </>
  );
}
