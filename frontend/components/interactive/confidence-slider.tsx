"use client";

import { Slider } from "@/components/ui/slider";
import { Card } from "@/components/ui/card";
import { fmt } from "@/lib/utils";

interface ConfidenceSliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  label?: string;
  presets?: number[];
}

const DEFAULT_PRESETS = [0.95, 0.99, 0.995, 0.999];

export function ConfidenceSlider({
  value,
  onChange,
  min = 0.9,
  max = 0.9999,
  step = 0.001,
  label = "置信水平",
  presets = DEFAULT_PRESETS,
}: ConfidenceSliderProps) {
  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-foreground">{label}</span>
        <span className="text-lg font-mono font-bold text-primary">
          {fmt(value * 100, 2)}%
        </span>
      </div>
      <Slider
        value={[value]}
        onValueChange={(v) => {
          const val = Array.isArray(v) ? v[0] : v;
          onChange(val);
        }}
        min={min}
        max={max}
        step={step}
        className="mb-3"
      />
      <div className="flex gap-2">
        {presets.map((p) => (
          <button
            key={p}
            onClick={() => onChange(p)}
            className={`px-3 py-1 text-xs font-mono rounded-md border transition-colors ${
              Math.abs(value - p) < 0.0001
                ? "bg-primary/10 border-primary/40 text-primary"
                : "border-border text-muted-foreground hover:border-primary/30 hover:text-foreground"
            }`}
          >
            {fmt(p * 100, 1)}%
          </button>
        ))}
      </div>
    </Card>
  );
}
