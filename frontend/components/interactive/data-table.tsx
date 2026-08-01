"use client";

import { useState, useMemo } from "react";
import type { ColumnStatistics } from "@/lib/types";
import { fmt } from "@/lib/utils";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { ArrowUpDown, Check, X } from "lucide-react";

interface DataTableProps {
  columns: ColumnStatistics[];
}

type SortKey = "column" | "mean" | "std" | "skew" | "kurtosis" | "min" | "max";

export function DataTable({ columns }: DataTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("column");
  const [sortAsc, setSortAsc] = useState(true);

  const sorted = useMemo(() => {
    const copy = [...columns];
    copy.sort((a, b) => {
      const va = a[sortKey];
      const vb = b[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === "string" && typeof vb === "string")
        return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
      return sortAsc ? (va as number) - (vb as number) : (vb as number) - (va as number);
    });
    return copy;
  }, [columns, sortKey, sortAsc]);

  function handleSort(key: SortKey) {
    if (key === sortKey) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(true); }
  }

  const SortHeader = ({ label, sortKeyName }: { label: string; sortKeyName: SortKey }) => (
    <button
      onClick={() => handleSort(sortKeyName)}
      className="inline-flex items-center gap-1 hover:text-foreground transition-colors"
    >
      {label}
      <ArrowUpDown className="h-3 w-3" />
    </button>
  );

  const columnLabels: Record<string, string> = {
    spread_all: "全品种",
    spread_5y: "5 年期",
    spread_10y: "10 年期",
    spread_30y: "30 年期",
  };

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead><SortHeader label="品种" sortKeyName="column" /></TableHead>
          <TableHead>N</TableHead>
          <TableHead><SortHeader label="均值" sortKeyName="mean" /></TableHead>
          <TableHead><SortHeader label="标准差" sortKeyName="std" /></TableHead>
          <TableHead><SortHeader label="偏度" sortKeyName="skew" /></TableHead>
          <TableHead><SortHeader label="峰度" sortKeyName="kurtosis" /></TableHead>
          <TableHead><SortHeader label="最小" sortKeyName="min" /></TableHead>
          <TableHead><SortHeader label="最大" sortKeyName="max" /></TableHead>
          <TableHead>ADF p</TableHead>
          <TableHead>平稳性</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {sorted.map((col) => (
          <TableRow key={col.column}>
            <TableCell className="font-medium">
              {columnLabels[col.column] ?? col.column}
            </TableCell>
            <TableCell className="font-mono text-muted-foreground">{col.n}</TableCell>
            <TableCell className="font-mono">{fmt(col.mean, 4)}</TableCell>
            <TableCell className="font-mono">{fmt(col.std, 4)}</TableCell>
            <TableCell className={cn("font-mono", Math.abs(col.skew) > 1 && "text-chart-3")}>
              {fmt(col.skew, 3)}
            </TableCell>
            <TableCell className={cn("font-mono", col.kurtosis > 3 && "text-chart-3")}>
              {fmt(col.kurtosis, 3)}
            </TableCell>
            <TableCell className="font-mono">{fmt(col.min, 4)}</TableCell>
            <TableCell className="font-mono">{fmt(col.max, 4)}</TableCell>
            <TableCell className="font-mono">
              {col.adf_pvalue != null ? fmt(col.adf_pvalue, 4) : "—"}
            </TableCell>
            <TableCell>
              {col.is_stationary != null ? (
                col.is_stationary ? (
                  <Badge variant="outline" className="text-chart-2 border-chart-2/30">
                    <Check className="h-3 w-3 mr-1" /> 平稳
                  </Badge>
                ) : (
                  <Badge variant="outline" className="text-chart-3 border-chart-3/30">
                    <X className="h-3 w-3 mr-1" /> 非平稳
                  </Badge>
                )
              ) : (
                <span className="text-muted-foreground">—</span>
              )}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
