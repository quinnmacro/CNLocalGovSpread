# 🤝 Handoff — CNLocalGovSpread v5.1 — 波动率页面现代化 (Phase 6 完成)

> **Generated**: 2026-08-02
> **Status**: Phase 6 Complete — All verification green ✅
> **Previous**: `HANDOFF-v5.0-volatility.md` (spec), `HANDOFF-v4.1.md` (Phase 1-5)

---

## 1. Phase 6 完成总结

### 1.1 新增 4 个后端模型

| 文件 | 模型 | 核心方法 | 行数 |
|------|------|---------|------|
| `src/models/har_rv.py` | HAR-RV (Corsi 2009) | 日/周/月三窗口 OLS | ~230 |
| `src/models/stochastic_vol.py` | Quasi-Bayesian SV | Kalman smoother + log-squared 变换 | ~460 |
| `src/models/gas_volatility.py` | GAS(1,1) (Creal 2013) | Score-driven QMLE (Normal/Student-t) | ~290 |
| `src/models/ms_garch.py` | MS-GARCH | HMM on |r| + regime-weighted GARCH | ~340 |

所有模型实现 `VolatilityModel` ABC (`fit`, `conditional_variance`, `forecast`, `diagnose`)。
`src/models/__init__.py` 已更新导出。

### 1.2 API 层 (4 个新端点)

| # | Endpoint | 响应时间 | 说明 |
|---|----------|---------|------|
| 22 | `GET /volatility/har-rv` | ~1.4s | HAR-RV 三窗口分解 |
| 23 | `GET /volatility/stochastic-vol` | ~28ms | 准贝叶斯 SV (Kalman smoother) |
| 24 | `GET /volatility/gas` | ~0.6s | GAS score-driven 波动率 |
| 25 | `GET /volatility/ms-garch` | ~0.5s | MS-GARCH 状态切换 |

5 个新 Pydantic schema: `HARRVResponse`, `StochasticVolResponse`, `GASResponse`, `MSRegimeInfo`, `MSGARCHResponse`

### 1.3 前端 (5 图表 + 页面重写)

**5 个新图表组件**:
- `har-rv-decomposition.tsx` — 3 线 (daily/weekly/monthly RV) + fitted σ²
- `stochastic-vol-band.tsx` — 后验均值 + 80% CI fill band
- `gas-score-dynamics.tsx` — 双面板: cond vol + score bars (green/red)
- `ms-garch-regimes.tsx` — scatter 按 regime 着色
- `volatility-model-comparison.tsx` — 全模型条件波动率叠加

**页面重写**: `volatility-content.tsx` 从 412 行 → 879 行
- `useScrollSpy` + 右侧 sticky TOC
- 5 段叙事结构: WHY / HOW / WHAT / SO WHAT / NOW WHAT
- KaTeX `<Formula>` 覆盖全部 8 个模型方程
- `<ReadGuide>` 每个图表
- `<ParamTooltip>` 关键参数
- `<PageNavigation>` 底部导航
- 回到顶部按钮
- Tournament 表扩展至 8-9 个模型

**基础设施**:
- `frontend/lib/types.ts` — 追加 5 个 TS interface
- `frontend/lib/api.ts` — 追加 4 个方法到 `api` 对象
- `frontend/hooks/use-api.ts` — 追加 4 个 hooks: `useHarRv`, `useStochasticVol`, `useGas`, `useMsGarch`

---

## 2. 验收红线 (All Green ✅)

| Check | Result |
|-------|--------|
| `npx tsc --noEmit` | ✅ 0 errors |
| `npx next build` | ✅ success (all 7 routes static) |
| `python3 -m pytest tests/ -q` | ✅ 53 passed (2.62s) |
| 4 new API endpoints | ✅ all HTTP 200 |

---

## 3. 关键技术决策

### 3.1 SV 模型: PyMC → Quasi-Bayesian Kalman Smoother

**问题**: PyMC ADVI 在 200 obs 上需要 60s+，不适合 web 端点实时响应。
**解决**: 实现快速准贝叶斯方法:
1. log(r² + c) 变换将 SV 转为线性状态空间
2. Yule-Walker 估计 AR(1) 参数 (μ, φ, σ_η)
3. Kalman forward filter + backward smoother 得到 h_t 路径
4. 80% CI 通过 exp((h ± 1.282·√P) / 2) 得到波动率区间

**结果**: 拟合时间从 60s+ 降至 2-3ms，同时保持与 PyMC 一致的后验均值 + CI 输出格式。
PyMC 全贝叶斯路径仍保留 (设置 `n_advi_steps > 5000` 可触发)。

### 3.2 ChartWrapper title prop

所有 `<ChartWrapper>` 必须传 `title` prop (required)。修复了 8 处缺失。

### 3.3 ModelSelector 接口

`ModelSelector` 只接受 `onFit` + `isFitting`，不接受 `models`/`selectedModel`/`onSelectModel`。已修正。

---

## 4. 项目当前完整清单

### 后端模型 (11 个)
```
src/models/
├── garch.py              # GARCH(1,1), EGARCH, GJR-GARCH
├── ewma.py               # EWMA (RiskMetrics)
├── figarch.py            # FIGARCH (长记忆)
├── kalman.py             # Kalman Filter
├── sts.py                # Structural Time Series
├── bayesian_sts.py       # Bayesian STS (PyMC)
├── har_rv.py             # HAR-RV ⭐ NEW
├── stochastic_vol.py     # Quasi-Bayesian SV ⭐ NEW
├── gas_volatility.py     # GAS(1,1) ⭐ NEW
├── ms_garch.py           # MS-GARCH ⭐ NEW
└── ml_volatility.py      # XGBoost/LightGBM (依赖未安装)
```

### API 端点 (25 个)
基础 7 + 进阶 10 + Regimes 4 + Volatility 4 = 25

### 前端页面 (6 个分析页 + 首页)
```
frontend/app/
├── page.tsx                    # 首页 Dashboard
├── analysis/
│   ├── overview/               # 市场概览
│   ├── volatility/             # 波动率分析 ⭐ REWRITTEN
│   ├── regimes/                # 状态识别
│   ├── risk/                   # 风险度量
│   └── scenarios/              # 情景分析
```

---

## 5. 文件变更清单 (未提交)

### Modified (5)
- `api/routes.py` — 追加 4 个端点 (#22-25)
- `api/schemas.py` — 追加 5 个 Pydantic schema
- `frontend/app/analysis/volatility/volatility-content.tsx` — 完整重写 (412→879 行)
- `frontend/hooks/use-api.ts` — 追加 4 个 hooks
- `src/models/__init__.py` — 追加 4 个模型导出

### New files (10)
- `src/models/har_rv.py`
- `src/models/stochastic_vol.py`
- `src/models/gas_volatility.py`
- `src/models/ms_garch.py`
- `frontend/components/charts/har-rv-decomposition.tsx`
- `frontend/components/charts/stochastic-vol-band.tsx`
- `frontend/components/charts/gas-score-dynamics.tsx`
- `frontend/components/charts/ms-garch-regimes.tsx`
- `frontend/components/charts/volatility-model-comparison.tsx`
- `HANDOFF-v5.0-volatility.md` (原始 spec)

### 注意: types.ts 和 api.ts 的变更已在之前的 session 中提交到 git

---

## 6. 可选后续工作

| 优先级 | 任务 | 说明 |
|--------|------|------|
| 🟡 可选 | 扩展至 ~1100 行 | 加 `<ExecutiveSummary>` + `<MetricCard>` 网格 + FIGARCH 详情 section |
| 🟡 可选 | SV 模型单元测试 | 为 4 个新模型添加测试 (当前 spec 禁止修改 tests) |
| 🟢 低 | Legacy dashboard 迁移 | `dashboard/` 旧 Dash 代码移入 `legacy/` |
| 🟢 低 | 新模型集成到 tournament | 将 HAR-RV/SV/GAS/MS-GARCH 加入 tournament 端点的自动对比 |

---

## 7. 依赖版本

```
numpy 2.4.6 | pandas 2.3.3 | scipy 1.15.3 | statsmodels 0.14.6
arch 8.0.0  | scikit-learn 1.8.0 | pymc 6.2.0 | hmmlearn 0.3.3
ruptures 1.1.10 | pytensor 3.2.3 | arviz 1.2.0
```

**注意**: 系统 Python 是 3.14 (缺 arch)，使用 `python3` (有全部依赖)。

---

**Handoff 完成**。下一个 session 可以从本文件 + `HANDOFF-v4.1.md` 无缝继续。
