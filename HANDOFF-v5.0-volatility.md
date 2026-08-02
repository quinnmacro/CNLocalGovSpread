# CNLocalGovSpread v5.0 — Phase 6: 波动率页面现代化升级

> **使用方法**: 将本 prompt + `HANDOFF-v4.1.md` 一起提供给下一个 session。
> **生成时间**: 2026-08-02
> **上一个 session 产出**: Phase 1–5 完成 (基础设施 + API + 5 分析页面 + Regimes 高级方法 + UI/UX polish)

---

## 你是谁

你是高级全栈工程师 + 量化金融专家，正在为 QuinnMacro 扩展波动率分析页面。Phase 1–5 已完成。你的任务是将 `/analysis/volatility` 页面从"教科书经典 GARCH 族"升级为"2020s 前沿实践"，同时保持与 regimes 页面一致的叙事深度和 UI 质量。

## 项目现状（已验证）

| 层级 | 状态 | 规模 | 备注 |
|------|------|------|------|
| `src/models/` | ✅ 7 个模型 | garch, ewma, figarch, ml_volatility, kalman, sts, bayesian_sts | **禁止修改现有文件** |
| `src/core/types.py` | ✅ 10 个 dataclass | 297 行 | 可新增类型 |
| `src/core/base.py` | ✅ 3 个 ABC | VolatilityModel, SignalExtractor, RiskAnalyzer | 可新增 ABC |
| `api/routes.py` | ✅ 21 个端点 | 1069 行 | **可修改（添加端点）** |
| `api/schemas.py` | ✅ 30 个 schema | 366 行 | **可修改（添加 schema）** |
| `frontend/` | ✅ 70+ TS/TSX 文件 | tsc 0 errors, next build ✅ | **可修改/新增** |
| `tests/` | ✅ 53 passed | 2.82s | **禁止修改** |

### 依赖版本（已验证安装）

```
numpy 2.4.6 | pandas 2.3.3 | scipy 1.15.3 | statsmodels 0.14.6
arch 8.0.0  | scikit-learn 1.8.0 | pymc 6.2.0 | hmmlearn 0.3.3
ruptures 1.1.10 | pytensor 3.2.3 | arviz 1.2.0
```

**注意**: xgboost 和 lightgbm **未安装**（ml_volatility.py 存在但当前无法运行，这不影响现有 53 测试）。

## 核心问题：为什么需要升级

当前波动率页面 (`volatility-content.tsx`, 412 行) 只用了以下模型：

| 模型 | 实现文件 | 年份 | 问题 |
|------|---------|------|------|
| GARCH(1,1) | `garch.py` (237L) | 1986 Bollerslev | 无法捕捉跳跃、长记忆、非对称尾部 |
| EGARCH | `garch.py` | 1991 Nelson | 杠杆效应建模粗糙，参数常触边界 |
| GJR-GARCH | `garch.py` | 1993 | 同上 |
| EWMA | `ewma.py` (223L) | 1994 RiskMetrics | 无均值回归，常数预测 |
| FIGARCH | `figarch.py` (372L) | 1996 Baillie | 长记忆参数 d 估计不稳定 |
| XGBoost/LightGBM | `ml_volatility.py` (360L) | 2016 | 特征工程粗糙，**依赖未安装** |

与 regimes 页面（1111 行，7 个模型 + useScrollSpy + PageNavigation + ExecutiveSummary）相比，波动率页面严重落后。

## 任务目标

### 1. 新增 4 个后端波动率模型

每个模型在 `src/models/` 下新建文件，**不修改任何现有 `src/models/*.py`**。
每个模型必须实现 `VolatilityModel` ABC（`fit`, `conditional_variance`, `forecast`, `diagnose`）。

#### a) HAR-RV — `src/models/har_rv.py`（必做）

```
Heterogeneous Autoregressive Realized Volatility (Corsi 2009)

RV_t = β_0 + β_d·RV_{t-1} + β_w·RV_{t-5} + β_m·RV_{t-22} + ε_t

实现要点：
- 输入: returns (日收益率序列)
- 计算已实现方差: RV_t = Σ_{i=t-4}^{t} r²_i (5日滚动平方和作为日度代理)
- 三个滞后窗口: 1d, 5d, 22d（对应日交易者/周交易者/月投资者）
- OLS 估计 (statsmodels 或手写 numpy)
- 输出 VolatilityResult + HAR 特有参数

新增类型 (在 src/core/types.py 添加):
@dataclass(frozen=True)
class HARRVResult(VolatilityResult):
    har_params: dict[str, float]  # β_0, β_d, β_w, β_m
    rv_daily: pd.Series           # 日已实现方差
    rv_5d: pd.Series              # 5 日滚动
    rv_22d: pd.Series             # 22 日滚动
    rv_66d: pd.Series             # 66 日滚动
    r_squared: float              # HAR 拟合 R²

参考模式: ewma.py (简单估计, VolatilityModel ABC 实现)
```

#### b) Stochastic Volatility — `src/models/stochastic_vol.py`（必做）

```
Bayesian Stochastic Volatility (Taylor 1986, Kim-Shephard-Chib 1998)

r_t = exp(h_t/2) · ε_t,    ε_t ~ N(0,1)
h_t = μ + φ·(h_{t-1} - μ) + η_t,  η_t ~ N(0, σ²_η)

实现要点：
- 使用 PyMC + ADVI（复用 bayesian_sts.py 的模式）
- 参数: μ (长期对数方差), φ (持续性), σ_η (波动率的波动率)
- 先验: μ ~ N(0, 10), φ ~ Beta(20, 1.5) [mapped to (-1,1)], σ_η ~ HalfNormal(0.5)
- 后验: h_t 的后验均值 + 80% HPD 区间
- 输出 VolatilityResult + 后验区间

新增类型 (在 src/core/types.py 添加):
@dataclass(frozen=True)
class StochasticVolResult(VolatilityResult):
    log_vol: pd.Series                # 后验均值对数波动率
    log_vol_lower: pd.Series          # 10th percentile
    log_vol_upper: pd.Series          # 90th percentile
    sv_params: dict[str, float]       # μ, φ, σ_η
    fitting_time_sec: float
    n_samples: int

依赖: pymc 6.2.0 + arviz 1.2.0（均已安装）
参考模式: bayesian_sts.py (PyMC + ADVI + 缓存)
```

#### c) GAS Volatility — `src/models/gas_volatility.py`（推荐）

```
Generalized Autoregressive Score (Creal, Koopman, Lucas 2013)
也称作 Observation-Driven / Score-Driven Model

f_{t+1} = ω + A·s_t + B·f_t
s_t = S_t · ∂log p(y_t|f_t)/∂f_t    (score of observation density)

当 f_t = σ²_t, y_t ~ N(0, f_t) 时，GAS(1,1) ≈ GARCH(1,1)
当使用 Student-t 分布时，score 自动降权异常值 → 更稳健

实现要点：
- GAS(1,1) with Normal 和 Student-t 两种分布
- 参数: ω (scalar), A (scalar), B (scalar), plus distribution params
- S_t = 1 (identity scaling) 或 S_t = I^{-1} (full information scaling)
- QMLE via scipy.optimize.minimize (参考 garch.py 的优化模式)
- 输出 VolatilityResult + score dynamics

新增类型 (在 src/core/types.py 添加):
@dataclass(frozen=True)
class GASResult(VolatilityResult):
    gas_params: dict[str, float]    # ω, A, B + distribution params
    score_dynamics: pd.Series       # s_t time series
    distribution: str               # "normal" or "studentst"
    scaling: str                    # "identity" or "full_info"

参考模式: garch.py (QMLE 框架) + figarch.py (自定义优化)
```

#### d) MS-GARCH — `src/models/ms_garch.py`（推荐）

```
Markov-Switching GARCH (Hamilton + Gruen 2012 type)

状态 s_t ∈ {1, ..., K} 服从马尔可夫链
r_t | s_t=k ~ N(μ_k, σ²_{k,t})
σ²_{k,t} = ω_k + α_k·ε²_{t-1} + β_k·σ²_{k,t-1}

实现要点：
- K=2 或 K=3 状态
- 简化实现: 先做 HMM 分状态 (hmmlearn), 然后各状态分别拟合 GARCH
  - 这不是严格的 MS-GARCH (path-dependent)，但是实用的近似
  - 严格实现需要 Hamilton filter + EM，复杂度较高
- 如果时间允许: 实现完整 Hamilton filter
- 否则: 使用 "分段 GARCH" 近似 → 结果也很实用

新增类型 (在 src/core/types.py 添加):
@dataclass(frozen=True)
class MSGARCHResult(VolatilityResult):
    regime_labels: np.ndarray
    regime_probs: pd.DataFrame          # (n_obs, K) 状态概率
    transition_matrix: np.ndarray
    regime_params: list[dict[str, float]]  # 各状态 GARCH 参数
    n_regimes: int

依赖: hmmlearn 0.3.3 + arch 8.0.0（均已安装）
参考模式: hmm_regime.py (HMM) + garch.py (GARCH fitting)
```

### 2. 扩展 API 端点

在 `api/routes.py` 添加 4 个新端点，在 `api/schemas.py` 添加对应 schema：

```python
# api/routes.py 新增:

# ---- 15. HAR-RV ----
@router.get("/volatility/har-rv", response_model=HARRVResponse)
async def har_rv(column: str = Query(...)):
    """HAR-RV realized volatility decomposition."""

# ---- 16. Stochastic Volatility ----
@router.get("/volatility/stochastic-vol", response_model=StochasticVolResponse)
async def stochastic_vol(column: str = Query(...)):
    """Bayesian stochastic volatility."""

# ---- 17. GAS Volatility ----
@router.get("/volatility/gas", response_model=GASResponse)
async def gas_volatility(column: str = Query(...)):
    """Score-driven GAS volatility model."""

# ---- 18. MS-GARCH ----
@router.get("/volatility/ms-garch", response_model=MSGARCHResponse)
async def ms_garch(column: str = Query(...)):
    """Markov-switching GARCH."""

# api/schemas.py 新增:

class HARRVResponse(BaseModel):
    model_name: str
    conditional_volatility: list[TimePoint]
    har_params: dict[str, float]     # β_0, β_d, β_w, β_m
    rv_daily: list[TimePoint]
    rv_5d: list[TimePoint]
    rv_22d: list[TimePoint]
    rv_66d: list[TimePoint]
    r_squared: float
    aic: float | None = None
    bic: float | None = None
    diagnostics: DiagnosticsInfo | None = None

class StochasticVolResponse(BaseModel):
    model_name: str
    conditional_volatility: list[TimePoint]     # posterior mean vol
    log_vol: list[TimePoint]                    # log volatility
    log_vol_lower: list[TimePoint]              # 10th percentile
    log_vol_upper: list[TimePoint]              # 90th percentile
    sv_params: dict[str, float]                 # μ, φ, σ_η
    fitting_time_sec: float
    n_samples: int

class GASResponse(BaseModel):
    model_name: str
    conditional_volatility: list[TimePoint]
    gas_params: dict[str, float]
    score_dynamics: list[TimePoint]
    distribution: str
    aic: float | None = None
    bic: float | None = None
    diagnostics: DiagnosticsInfo | None = None

class MSGARCHResponse(BaseModel):
    model_name: str
    conditional_volatility: list[TimePoint]
    regime_labels: list[int]
    regime_probabilities: list[dict[str, float]]
    transition_matrix: list[list[float]]
    regime_params: list[dict[str, float]]
    n_regimes: int
    aic: float | None = None
    bic: float | None = None
```

**关键**: 每个端点的 `_get_returns()` 逻辑已在 routes.py 中存在（`_get_returns()` helper），直接复用。

### 3. 前端升级

#### 3.1 新增 Types（`frontend/lib/types.ts`）

```typescript
export interface HARRVResponse {
  model_name: string;
  conditional_volatility: TimePoint[];
  har_params: Record<string, number>;
  rv_daily: TimePoint[];
  rv_5d: TimePoint[];
  rv_22d: TimePoint[];
  rv_66d: TimePoint[];
  r_squared: number;
  aic: number | null;
  bic: number | null;
  diagnostics: DiagnosticsInfo | null;
}

export interface StochasticVolResponse {
  model_name: string;
  conditional_volatility: TimePoint[];
  log_vol: TimePoint[];
  log_vol_lower: TimePoint[];
  log_vol_upper: TimePoint[];
  sv_params: Record<string, number>;
  fitting_time_sec: number;
  n_samples: number;
}

export interface GASResponse {
  model_name: string;
  conditional_volatility: TimePoint[];
  gas_params: Record<string, number>;
  score_dynamics: TimePoint[];
  distribution: string;
  aic: number | null;
  bic: number | null;
  diagnostics: DiagnosticsInfo | null;
}

export interface MSGARCHResponse {
  model_name: string;
  conditional_volatility: TimePoint[];
  regime_labels: number[];
  regime_probabilities: Array<Record<string, number>>;
  transition_matrix: number[][];
  regime_params: Array<Record<string, number>>;
  n_regimes: number;
  aic: number | null;
  bic: number | null;
}
```

#### 3.2 新增 API Client（`frontend/lib/api.ts`）

```typescript
// 在 api 对象中添加:
harRV: () => fetchApi<HARRVResponse>("/volatility/har-rv"),
stochasticVol: () => fetchApi<StochasticVolResponse>("/volatility/stochastic-vol"),
gasVolatility: () => fetchApi<GASResponse>("/volatility/gas"),
msGarch: () => fetchApi<MSGARCHResponse>("/volatility/ms-garch"),
```

#### 3.3 新增 Hooks（`frontend/hooks/use-api.ts`）

```typescript
export function useHARRV(opts?: ...) { ... }
export function useStochasticVol(opts?: ...) { ... }
export function useGASVolatility(opts?: ...) { ... }
export function useMSGARCH(opts?: ...) { ... }
```

#### 3.4 新增 5 个图表组件（`frontend/components/charts/`）

1. **`har-rv-decomposition.tsx`** — HAR 三成分分解
   - 上: 日 RV + 5d + 22d 三条线（多时间尺度）
   - 下: HAR 残差（应近似白噪声）
   - 参考: `sts-signal.tsx` 双 panel + 事件标注

2. **`stochastic-vol-band.tsx`** — SV 后验波动率带
   - 后验均值波动率线
   - 80% HPD 区间填充（半透明带）
   - 与 GARCH 波动率叠加对比
   - 参考: `bayesian-sts.tsx` 后验带图

3. **`gas-score-dynamics.tsx`** — GAS Score 动态
   - Score 时间序列 + 条件波动率双轴
   - 标注 |score| > 2σ 的异常点
   - 参考: `kalman-signal.tsx` 双 panel

4. **`ms-garch-regimes.tsx`** — MS-GARCH 状态叠加
   - 条件波动率线（颜色按状态变化）
   - 背景色块标注各状态概率
   - 参考: `regime-sequence.tsx`

5. **`volatility-model-comparison.tsx`** — 多模型对比面板
   - 所有模型条件波动率叠加
   - 评估指标: RMSE, MAE, QLIKE
   - 参考: `volatility-overlay.tsx`（扩展）

#### 3.5 重写 `volatility-content.tsx`

从当前的 5 段 412 行升级为 **7 段 ~1100 行**，参考 regimes-content.tsx 结构：

```
TOC_SECTIONS = [
  { id: "why", label: "WHY — 研究动机" },
  { id: "how", label: "HOW — 方法论" },
  { id: "what", label: "WHAT — 模型结果" },
  { id: "so-what", label: "SO WHAT — 诊断选择" },
  { id: "now-what", label: "NOW WHAT — 投资应用" },
  { id: "cross-validation", label: "交叉验证" },
  { id: "advanced", label: "进阶方法" },
]

关键改进:
- 添加 useScrollSpy + 右侧 TOC 导航
- 添加 ExecutiveSummary 组件
- 添加 PageNavigation 组件
- 每个新模型有 KaTeX 公式 + ReadGuide + ParamTooltip
- 与 regimes 页形成交叉引用 (Link)
```

**Section 结构详细规划**:

**Section 0: WHY — 为什么经典 GARCH 不够用**
```
- GARCH 的三大局限:
  1. 单指数衰减记忆（vs HAR 的多时间尺度）
  2. 确定性方差函数（vs SV 的随机波动率）
  3. 单一参数集（vs MS-GARCH 的状态依赖）
- 数据现实: 利差序列有跳跃、长记忆、状态切换
- Formula: GARCH(1,1) 递推 + 标注局限性
```

**Section 1: HOW — 现代波动率方法论**
```
4 个新方法的公式 + 经济学解释:

HAR-RV:
  Formula: RV_t = β_0 + β_d·RV_{t-1} + β_w·RV_{t-5} + β_m·RV_{t-22} + ε_t
  经济学: 市场参与者异质性 → 多时间尺度波动率成分

SV:
  Formula: r_t = exp(h_t/2)·ε_t; h_t = μ + φ(h_{t-1}-μ) + η_t
  经济学: 波动率有自己的随机驱动 → "波动率的波动率"

GAS:
  Formula: f_{t+1} = ω + A·s_t + B·f_t; s_t = S_t·∇log p(y_t|f_t)
  经济学: Score-driven 更新 → 自动降权异常值 → 稳健

MS-GARCH:
  Formula: σ²_{k,t} = ω_k + α_k·ε²_{t-1} + β_k·σ²_{k,t-1}
  经济学: 不同市场状态下波动率动态不同
```

**Section 2: WHAT — 模型结果展示**
```
- Tournament 表扩展（8-9 模型: GARCH, EGARCH, GJR, EWMA, FIGARCH, HAR-RV, SV, GAS, MS-GARCH）
- 条件波动率叠加图（所有模型）
- HAR 分解图
- SV 后验带图
- GAS score 动态图
- MS-GARCH 状态叠加图
```

**Section 3: SO WHAT — 模型诊断与选择**
```
- 残差诊断 ACF / QQ / ARCH-LM（现有）
- 预测性能对比表: RMSE, MAE, QLIKE
- 参数稳定性分析
- 方法论对比表（记忆结构/不确定性/跳跃/计算复杂度/解释性）
```

**Section 4: NOW WHAT — 投资应用**
```
- 波动率择时信号
- 风险度量 VaR/ES 的时变输入
- 压力测试场景生成
- 与 regimes 页联合: 高波 + 状态转换 → 强防御
```

**Section 5: 交叉验证 — 与 regimes 页联动**
```
- HMM 状态 vs MS-GARCH 状态一致性
- Kalman 信号 vs 条件波动率相关性
- 联合信号解读
```

**Section 6: 进阶 — 方法论对比表**
```
- 8+ 模型对比矩阵
- 模型选择决策树
```

### 4. 设计哲学（不可违背）

1. **每个新方法必须有 KaTeX 公式 + 经济学解释**
2. **每个新图表必须有 `<ReadGuide>` 读图指南**
3. **关键参数必须有 `<ParamTooltip>` 经济学 tooltip**
4. **中文为主，技术术语括号加英文**
5. **与现有 GARCH/FIGARCH 形成互补叙事（不是替代，是扩展）**
6. **Tournament 表包含所有模型**（经典 vs 现代对比）

## 技术约束

1. **不要碰 `src/models/` 和 `tests/` 中的现有文件** — 53 测试不能破坏
2. **可以新增 `src/models/*.py` 文件** — 参考 `sts.py` / `bayesian_sts.py` 模式
3. **可以修改 `api/routes.py` 和 `api/schemas.py`** — 添加新端点
4. **可以修改 `src/core/types.py`** — 添加新结果类型
5. **新增 Python 代码必须通过 `pytest tests/ -v`** — 仍为 53 passed
6. **TypeScript strict，不允许 `any`**
7. **深色主题** — 使用 `globals.css` 中已定义的 CSS 变量
8. **shadcn/ui 使用 `@base-ui/react`**
9. **Plotly 必须用 `next/dynamic` (ssr: false)** — 参考 `plotly-chart.tsx`
10. **KaTeX** — 使用 `Formula` 组件，`block=true` 用于 display math
11. **Section 组件支持 `id` 属性** — 用于 useScrollSpy 定位

## 运行命令

```bash
# 后端
cd /Users/liulu/Code/CNLocalGovSpread
CLS_DATA__SOURCE=mock python3.13 -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload

# 前端
cd /Users/liulu/Code/CNLocalGovSpread/frontend
npm run dev  # http://localhost:3000

# 验证
cd /Users/liulu/Code/CNLocalGovSpread/frontend
npx tsc --noEmit  # 必须 0 errors
npx next build     # 必须成功

# 测试
cd /Users/liulu/Code/CNLocalGovSpread
python3.13 -m pytest tests/ -v  # 必须 53 passed（不能减少）
```

## 验收标准

1. ✅ `npx tsc --noEmit` 零错误
2. ✅ `npx next build` 成功
3. ✅ `python3.13 -m pytest tests/ -v` 53 passed（不减少）
4. ✅ 4 个新波动率模型实现（HAR-RV, SV, GAS, MS-GARCH）
5. ✅ 4 个新 API 端点可用
6. ✅ `/analysis/volatility` 页面升级完成（~1100 行，7 段结构）
7. ✅ 每个新方法有 KaTeX 公式 + ReadGuide + ParamTooltip
8. ✅ 与 regimes 页形成交叉验证叙事
9. ✅ 新增 5 个图表组件
10. ✅ Tournament 表扩展至 8-9 个模型
11. ✅ 添加 `useScrollSpy` 支持 TOC 导航
12. ✅ 添加 `PageNavigation` + `ExecutiveSummary` 组件

## 文件参考（精确路径 + 行数）

### 后端模式参考
| 参考文件 | 行数 | 学习什么 |
|---------|------|---------|
| `src/models/ewma.py` | 223 | VolatilityModel ABC + 简单估计 + forecast |
| `src/models/garch.py` | 237 | QMLE 拟合 + arch 库 + diagnostics |
| `src/models/bayesian_sts.py` | 233 | PyMC + ADVI + 结果缓存模式 |
| `src/models/sts.py` | 181 | statsmodels UnobservedComponents |
| `src/models/figarch.py` | 372 | 自定义 scipy.optimize + GPH |
| `src/regime/hmm_regime.py` | — | EM algorithm + hmmlearn |
| `src/core/types.py` | 297 | 所有结果 dataclass 定义 |
| `src/core/base.py` | — | VolatilityModel ABC 接口 |

### 前端模式参考
| 参考文件 | 行数 | 学习什么 |
|---------|------|---------|
| `frontend/app/analysis/regimes/regimes-content.tsx` | 1111 | 7 段结构 + useScrollSpy + TOC |
| `frontend/components/charts/bayesian-sts.tsx` | — | 后验带图 (CI 填充) |
| `frontend/components/charts/sts-signal.tsx` | — | 双 panel + 事件标注 |
| `frontend/components/charts/kalman-signal.tsx` | — | z-score + 阈值 + 背景色 |
| `frontend/components/charts/regime-sequence.tsx` | — | 状态序列 + 概率色块 |
| `frontend/components/charts/volatility-overlay.tsx` | 60 | 多模型波动率叠加 |
| `frontend/components/charts/tournament-table.tsx` | 113 | 可排序模型对比表 |
| `frontend/hooks/use-scroll-spy.ts` | 57 | ScrollSpy hook |
| `frontend/components/narrative/page-navigation.tsx` | — | 上下页导航 |
| `frontend/components/narrative/executive-summary.tsx` | — | 页面摘要卡片 |
| `frontend/components/narrative/formula.tsx` | — | KaTeX 公式 (block/inline) |
| `frontend/components/narrative/read-guide.tsx` | — | 读图指南 |
| `frontend/components/narrative/param-tooltip.tsx` | — | 参数 tooltip |

### API 模式参考
| 参考文件 | 行数 | 学习什么 |
|---------|------|---------|
| `api/routes.py` | 1069 | 端点定义 + _series_to_timepoints helper |
| `api/schemas.py` | 366 | Pydantic schema 定义 |
| `frontend/hooks/use-api.ts` | 256 | TanStack Query hooks |
| `frontend/lib/api.ts` | 124 | fetchApi 封装 |
| `frontend/lib/types.ts` | 340 | TypeScript 类型 |

## 实现顺序建议

```
Phase 6A: 后端模型 (可并行)
├── 1. HAR-RV (最简单, OLS, ~2h)
├── 2. Stochastic Volatility (PyMC, ~3h)
├── 3. GAS (scipy.optimize, ~3h)
└── 4. MS-GARCH (HMM + GARCH, ~4h)

Phase 6B: API + Schema (~2h)
├── api/schemas.py 添加 4 个 Response
├── api/routes.py 添加 4 个端点
└── 验证端点可用 (curl / httpie)

Phase 6C: 前端基础设施 (~1h)
├── types.ts 添加 4 个接口
├── api.ts 添加 4 个 client
└── use-api.ts 添加 4 个 hooks

Phase 6D: 图表组件 (可并行, ~4h)
├── har-rv-decomposition.tsx
├── stochastic-vol-band.tsx
├── gas-score-dynamics.tsx
├── ms-garch-regimes.tsx
└── volatility-model-comparison.tsx

Phase 6E: 页面组装 (~3h)
├── volatility-content.tsx 重写
├── useScrollSpy + TOC
├── KaTeX 公式 + ReadGuide + ParamTooltip
├── ExecutiveSummary + PageNavigation
└── 交叉验证 section

Phase 6F: 验证 (~1h)
├── npx tsc --noEmit
├── npx next build
├── pytest tests/ -v
└── 手动检查所有新端点 + 图表
```

## 经济学叙事参考

### 方法论对比矩阵

| 维度 | GARCH(1,1) | HAR-RV | Stochastic Vol | GAS(1,1) | MS-GARCH |
|------|-----------|--------|----------------|----------|----------|
| 年份 | 1986 | 2009 | 1986/1998 | 2013 | 2012 |
| 记忆结构 | 单指数衰减 | 三窗口异质 | 潜在 AR(1) | Score-driven | 状态切换 |
| 不确定性 | 无(点估计) | 无(OLS) | 有(后验) | 有(Fisher info) | 有(状态概率) |
| 跳跃捕捉 | 差 | 中(RV包含) | 好(可扩展) | 中(score稳健) | 好(高波状态) |
| 计算复杂度 | 低 | 极低 | 高(MCMC/ADVI) | 低(QMLE) | 中(EM) |
| 解释性 | 高 | 高 | 中 | 中 | 高 |

### HAR-RV 叙事

> 市场参与者有不同的时间尺度：日内交易者关注隔夜波动，周交易者关注周度趋势，
> 月度投资者关注宏观周期。HAR-RV 用三个滞后窗口 (1d, 5d, 22d) 分别捕捉
> 这三种时间尺度的波动率成分。比 GARCH 的"单一记忆"更符合市场微观结构。

### SV 叙事

> GARCH 假设波动率是过去冲击的确定性函数——给定历史信息，今天的方差完全已知。
> 但现实中，波动率有自己的随机驱动（央行政策不确定性、地缘政治事件等），
> 不完全由过去的收益率决定。SV 让波动率成为独立的随机过程，
> 贝叶斯框架给出参数的完整后验分布——不只是"σ=0.02"，而是"σ 有 80% 概率在 [0.015, 0.028]"。

### GAS 叙事

> GAS 是 GARCH 的理论泛化。GARCH 的更新规则 σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
> 恰好等于 GAS 在正态分布假设下的特例。当使用 Student-t 分布时，
> GAS 的 score 函数自动对异常值降权——这比 GARCH 的 ε² 更新更稳健，
> 不会因为一个极端收益率就大幅抬升未来的波动率预测。

### MS-GARCH 叙事

> regimes 页面已经识别了市场状态（低波/中波/高波）。MS-GARCH 进一步让
> 波动率模型本身也随状态变化：在"平静期"用低持久性的 GARCH 参数，
> 在"危机期"用高持久性的参数。这比单一 GARCH 更好地解释了为什么
> 波动率聚集不是"一次冲击慢慢衰减"，而是"不同状态之间的切换"。

---

**Handoff 完成**。将此 prompt + `HANDOFF-v4.1.md` 提供给下一个 session，即可无缝启动 Phase 6。
