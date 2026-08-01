/**
 * TanStack Query hooks for all API endpoints.
 * Centralized data fetching with caching, refetch, and error handling.
 */

import { useQuery, useMutation, type UseQueryOptions } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type * as T from "@/lib/types";

// ─── Query keys ─────────────────────────────────────────────────────

export const queryKeys = {
  health: ["health"] as const,
  dataSummary: ["data", "summary"] as const,
  dataRaw: (limit: number, offset: number) => ["data", "raw", limit, offset] as const,
  dataStatistics: ["data", "statistics"] as const,
  fitModel: (type: string) => ["models", "fit", type] as const,
  tournament: ["models", "tournament"] as const,
  modelDetail: (name: string) => ["models", "detail", name] as const,
  figarch: ["models", "figarch"] as const,
  riskMetrics: (confidence: number) => ["risk", "metrics", confidence] as const,
  evt: (percentile: number) => ["risk", "evt", percentile] as const,
  backtest: (confidence: number) => ["risk", "backtest", confidence] as const,
  hmm: (n: number) => ["regimes", "hmm", n] as const,
  marketGauge: ["market", "gauge"] as const,
  scenarios: (horizon: number, nPaths: number) =>
    ["scenarios", "generate", horizon, nPaths] as const,
  sensitivity: ["analysis", "sensitivity"] as const,
  kalmanSignal: (column: string) => ["regimes", "kalman", column] as const,
  changepoints: (column: string, method: string) =>
    ["regimes", "changepoints", column, method] as const,
} as const;

// ─── Data hooks ─────────────────────────────────────────────────────

export function useHealth(opts?: Omit<UseQueryOptions<T.HealthResponse>, "queryKey" | "queryFn">) {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: api.health,
    staleTime: 5 * 60 * 1000,
    ...opts,
  });
}

export function useDataSummary(opts?: Omit<UseQueryOptions<T.DataSummary>, "queryKey" | "queryFn">) {
  return useQuery({
    queryKey: queryKeys.dataSummary,
    queryFn: api.dataSummary,
    staleTime: 10 * 60 * 1000,
    ...opts,
  });
}

export function useDataStatistics(opts?: Omit<UseQueryOptions<T.DataStatisticsResponse>, "queryKey" | "queryFn">) {
  return useQuery({
    queryKey: queryKeys.dataStatistics,
    queryFn: api.dataStatistics,
    staleTime: 10 * 60 * 1000,
    ...opts,
  });
}

// ─── Model hooks ────────────────────────────────────────────────────

export function useFitModel(
  modelType: string,
  opts?: Omit<UseQueryOptions<T.ModelResult>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.fitModel(modelType),
    queryFn: () => api.fitModel(modelType),
    staleTime: 5 * 60 * 1000,
    ...opts,
  });
}

export function useTournament(opts?: Omit<UseQueryOptions<T.TournamentResponse>, "queryKey" | "queryFn">) {
  return useQuery({
    queryKey: queryKeys.tournament,
    queryFn: api.tournament,
    staleTime: 5 * 60 * 1000,
    ...opts,
  });
}

export function useModelDetail(
  name: string,
  opts?: Omit<UseQueryOptions<T.ModelDetailResponse>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.modelDetail(name),
    queryFn: () => api.modelDetail(name),
    staleTime: 5 * 60 * 1000,
    ...opts,
  });
}

export function useFigarch(opts?: Omit<UseQueryOptions<T.FigarchResponse>, "queryKey" | "queryFn">) {
  return useQuery({
    queryKey: queryKeys.figarch,
    queryFn: api.figarch,
    staleTime: 5 * 60 * 1000,
    ...opts,
  });
}

export function useFitCustom() {
  return useMutation({
    mutationFn: (body: T.CustomFitRequest) => api.fitCustom(body),
  });
}

// ─── Risk hooks ────────────────────────────────────────────────────

export function useRiskMetrics(
  confidence = 0.99,
  opts?: Omit<UseQueryOptions<T.RiskMetrics>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.riskMetrics(confidence),
    queryFn: () => api.riskMetrics(confidence),
    staleTime: 5 * 60 * 1000,
    ...opts,
  });
}

export function useEvt(
  percentile = 0.1,
  opts?: Omit<UseQueryOptions<T.EvtResponse>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.evt(percentile),
    queryFn: () => api.evt(percentile),
    staleTime: 5 * 60 * 1000,
    ...opts,
  });
}

export function useBacktest(
  confidence = 0.99,
  opts?: Omit<UseQueryOptions<T.BacktestResponse>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.backtest(confidence),
    queryFn: () => api.backtest(confidence),
    staleTime: 5 * 60 * 1000,
    ...opts,
  });
}

// ─── Regime hooks ───────────────────────────────────────────────────

export function useHmm(
  nRegimes = 3,
  opts?: Omit<UseQueryOptions<T.HmmResponse>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.hmm(nRegimes),
    queryFn: () => api.hmm(nRegimes),
    staleTime: 5 * 60 * 1000,
    ...opts,
  });
}

export function useMarketGauge(opts?: Omit<UseQueryOptions<T.MarketGaugeResponse>, "queryKey" | "queryFn">) {
  return useQuery({
    queryKey: queryKeys.marketGauge,
    queryFn: api.marketGauge,
    staleTime: 2 * 60 * 1000,
    ...opts,
  });
}

// ─── Scenario hooks ─────────────────────────────────────────────────

export function useScenarios(
  horizon = 252,
  nPaths = 5000,
  opts?: Omit<UseQueryOptions<T.ScenarioResponse>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.scenarios(horizon, nPaths),
    queryFn: () => api.generateScenarios(horizon, nPaths),
    staleTime: 5 * 60 * 1000,
    ...opts,
  });
}

export function useStress() {
  return useMutation({
    mutationFn: (body: T.StressRequest) => api.stress(body),
  });
}

// ─── Analysis hooks ─────────────────────────────────────────────────

export function useSensitivity(opts?: Omit<UseQueryOptions<T.SensitivityResponse>, "queryKey" | "queryFn">) {
  return useQuery({
    queryKey: queryKeys.sensitivity,
    queryFn: api.sensitivity,
    staleTime: 5 * 60 * 1000,
    ...opts,
  });
}

// ─── Regime advanced hooks ─────────────────────────────────────────

export function useKalmanSignal(
  column = "spread_all",
  opts?: Omit<UseQueryOptions<T.KalmanSignalResponse>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.kalmanSignal(column),
    queryFn: () => api.kalmanSignal(column),
    staleTime: 5 * 60 * 1000,
    ...opts,
  });
}

export function useChangepoints(
  column = "spread_all",
  method = "binseg",
  nBkps = 5,
  opts?: Omit<UseQueryOptions<T.ChangepointResponse>, "queryKey" | "queryFn">
) {
  return useQuery({
    queryKey: queryKeys.changepoints(column, method),
    queryFn: () => api.changepoints(column, method, nBkps),
    staleTime: 5 * 60 * 1000,
    ...opts,
  });
}
