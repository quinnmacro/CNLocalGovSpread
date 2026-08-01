"use client";

import { Slider } from "@/components/ui/slider";
import { Card } from "@/components/ui/card";

interface HorizonSliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  presets?: number[];
}

const DEFAULT_PRESETS = [
  { value: 30, label: "1月" },
  { value: 60, label: "3月" },
  { value: 120, label: "6月" },
  { value: 252, label: "1年" },
];

export function HorizonSlider({
  value,
  onChange,
  min = 10,
  max = 500,
  step = 1,
  presets = DEFAULT_PRESETS,
}: HorizonSliderProps) {
  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-foreground">预测期限</span>
        <span className="text-lg font-mono font-bold text-primary">
          {value} 天
        </span>
      </div>
      <Slider
        value={[value]}
        onValueChange={(v: number[]) => onChange(v[0])}
        min={min}
        max={max}
        step={step}
        className="mb-3"
      />
      <div className="flex gap-2">
        {presets.map((p) => (
          <button
            key={p.value}
            onClick={() => onChange(p.value)}
            className={`px-3 py-1 text-xs font-mono rounded-md border transition-colors ${
              value === p.value
                ? "bg-primary/10 border-primary/40 text-primary"
                : "border-border text-muted-foreground hover:border-primary/30 hover:text-foreground"
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>
    </Card>
  );
}
