"use client";

import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { RefreshCw } from "lucide-react";
import type { CustomFitRequest } from "@/lib/types";

interface ModelSelectorProps {
  onFit: (request: CustomFitRequest) => void;
  isFitting?: boolean;
}

const MODEL_TYPES = [
  { value: "garch" as const, label: "GARCH" },
  { value: "egarch" as const, label: "EGARCH" },
  { value: "gjr" as const, label: "GJR-GARCH" },
];

const DISTRIBUTIONS = [
  { value: "normal" as const, label: "正态分布" },
  { value: "studentst" as const, label: "t 分布" },
  { value: "skewt" as const, label: "偏 t 分布" },
];

export function ModelSelector({ onFit, isFitting }: ModelSelectorProps) {
  const [modelType, setModelType] = useState<CustomFitRequest["model_type"]>("garch");
  const [dist, setDist] = useState<CustomFitRequest["dist"]>("studentst");

  function handleFit() {
    onFit({ model_type: modelType, p: 1, q: 1, dist });
  }

  return (
    <Card className="p-4">
      <h4 className="text-sm font-medium text-foreground mb-3">自定义模型拟合</h4>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div>
          <label className="text-xs text-muted-foreground mb-1 block">模型类型</label>
          <Select value={modelType} onValueChange={(v) => setModelType(v as CustomFitRequest["model_type"])}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="选择模型" />
            </SelectTrigger>
            <SelectContent>
              {MODEL_TYPES.map((m) => (
                <SelectItem key={m.value} value={m.value}>
                  {m.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div>
          <label className="text-xs text-muted-foreground mb-1 block">残差分布</label>
          <Select value={dist} onValueChange={(v) => setDist(v as CustomFitRequest["dist"])}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="选择分布" />
            </SelectTrigger>
            <SelectContent>
              {DISTRIBUTIONS.map((d) => (
                <SelectItem key={d.value} value={d.value}>
                  {d.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-end">
          <Button
            onClick={handleFit}
            disabled={isFitting}
            className="w-full"
            size="sm"
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isFitting ? "animate-spin" : ""}`} />
            {isFitting ? "拟合中..." : "重新拟合"}
          </Button>
        </div>
      </div>
    </Card>
  );
}
