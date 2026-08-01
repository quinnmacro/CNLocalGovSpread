"use client";
import { useState, useMemo } from "react";
import type { TournamentRow } from "@/lib/types";
import { fmt } from "@/lib/utils";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { ArrowUpDown, Check, X } from "lucide-react";

type SortKey = "model_name" | "aic" | "bic" | "persistence";

interface TournamentTableProps {
  models: TournamentRow[];
  winnerAic: string | null;
  winnerBic: string | null;
  onModelClick?: (name: string) => void;
  selectedModel?: string;
}

export function TournamentTable({
  models,
  winnerAic,
  winnerBic,
  onModelClick,
  selectedModel,
}: TournamentTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("aic");
  const [sortAsc, setSortAsc] = useState(true);

  const sorted = useMemo(() => {
    const copy = [...models];
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
  }, [models, sortKey, sortAsc]);

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

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead><SortHeader label="模型" sortKeyName="model_name" /></TableHead>
          <TableHead>类型</TableHead>
          <TableHead><SortHeader label="AIC" sortKeyName="aic" /></TableHead>
          <TableHead><SortHeader label="BIC" sortKeyName="bic" /></TableHead>
          <TableHead><SortHeader label="持续性" sortKeyName="persistence" /></TableHead>
          <TableHead>收敛</TableHead>
          <TableHead>ARCH</TableHead>
          <TableHead>正态</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {sorted.map((m) => {
          const isWinner = m.model_name === winnerAic || m.model_name === winnerBic;
          const isSelected = m.model_name === selectedModel;
          return (
            <TableRow
              key={m.model_name}
              className={cn(
                "cursor-pointer transition-colors",
                isSelected && "bg-primary/10",
                isWinner && "border-l-2 border-l-primary"
              )}
              onClick={() => onModelClick?.(m.model_name)}
            >
              <TableCell className="font-medium">
                <div className="flex items-center gap-2">
                  {m.model_name}
                  {isWinner && (
                    <Badge variant="outline" className="text-xs text-primary border-primary/40">
                      {m.model_name === winnerAic ? "AIC ✓" : ""}{m.model_name === winnerBic ? " BIC ✓" : ""}
                    </Badge>
                  )}
                </div>
              </TableCell>
              <TableCell className="text-muted-foreground text-xs font-mono">{m.model_type}</TableCell>
              <TableCell className="font-mono">{fmt(m.aic, 1)}</TableCell>
              <TableCell className="font-mono">{fmt(m.bic, 1)}</TableCell>
              <TableCell className="font-mono">{fmt(m.persistence, 4)}</TableCell>
              <TableCell>{m.converged ? <Check className="h-4 w-4 text-chart-2" /> : <X className="h-4 w-4 text-destructive" />}</TableCell>
              <TableCell>{m.has_arch_effects ? <X className="h-4 w-4 text-chart-3" /> : <Check className="h-4 w-4 text-chart-2" />}</TableCell>
              <TableCell>{m.is_normal ? <Check className="h-4 w-4 text-chart-2" /> : <X className="h-4 w-4 text-chart-3" />}</TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}
