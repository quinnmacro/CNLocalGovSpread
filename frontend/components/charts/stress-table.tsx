"use client";
import type { StressScenario } from "@/lib/types";
import { fmt, fmtPct } from "@/lib/utils";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface StressTableProps {
  scenarios: StressScenario[];
}

export function StressTable({ scenarios }: StressTableProps) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>情景</TableHead>
          <TableHead>波动率倍数</TableHead>
          <TableHead>中位数终值</TableHead>
          <TableHead>5% 分位</TableHead>
          <TableHead>95% 分位</TableHead>
          <TableHead>超标概率</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {scenarios.map((s) => (
          <TableRow key={s.name} className={cn(s.vol_multiplier >= 2 && "bg-destructive/5")}>
            <TableCell className="font-medium">{s.name}</TableCell>
            <TableCell className="font-mono">{fmt(s.vol_multiplier, 1)}x</TableCell>
            <TableCell className="font-mono">{fmt(s.median_final, 2)} bps</TableCell>
            <TableCell className="font-mono">{fmt(s.p5, 2)} bps</TableCell>
            <TableCell className="font-mono">{fmt(s.p95, 2)} bps</TableCell>
            <TableCell>
              <Badge
                variant="outline"
                className={cn(
                  "font-mono",
                  s.prob_exceed > 0.1 ? "border-destructive/50 text-destructive" :
                  s.prob_exceed > 0.05 ? "border-chart-3/50 text-chart-3" :
                  "border-chart-2/50 text-chart-2"
                )}
              >
                {fmtPct(s.prob_exceed)}
              </Badge>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
