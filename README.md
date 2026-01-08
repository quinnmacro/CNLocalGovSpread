# 中国地方政府债券利差高级计量经济学框架
## Advanced Econometric Framework for China Local Government Bond Spread Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 项目概述 (Project Overview)

本项目实现了一套**专业级的计量经济学框架**，用于分析中国地方政府债券利差的动态特征。核心理念是**"模型锦标赛"**（Model Tournament）——不依赖单一模型，而是让多个模型竞争，由数据驱动选择最优模型。

### 核心功能

✅ **三大分析模块**:
1. **波动率建模锦标赛** - GARCH/EGARCH/GJR-GARCH 模型竞争
2. **卡尔曼滤波器** - 从市场噪音中提取真实信号
3. **极值理论 (EVT)** - 基于 GPD 的尾部风险量化

✅ **交互式可视化** - 使用 Plotly 生成专业图表（中文标注）

✅ **战略输出报告** - 自动生成可执行的交易建议和风险预警

---

## 🏗️ 技术架构

### 模型实现细节

| 模块 | 技术 | 核心库 | 用途 |
|------|------|--------|------|
| **波动率建模** | GARCH 模型族 | `arch` | 捕捉波动率聚集和不对称效应 |
| **信号提取** | 卡尔曼滤波器 | `statsmodels` | 分离基本面信号与市场噪音 |
| **尾部风险** | 极值理论 (POT) | `scipy.stats` | 估计极端损失的概率分布 |
| **可视化** | 交互式图表 | `plotly` | 专业级数据展示 |

### 数据引擎

- **生产环境**: 支持 Wind EDB 数据接口（需配置 Wind API）
- **开发/测试**: 自带模拟数据生成器（GARCH 过程 + 跳跃扩散）
- **数据清洗**: 使用 MAD（中位数绝对偏差）方法处理异常值

---

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行分析

```bash
jupyter notebook china_localgov_bond_spread_analysis.ipynb
```

或直接在 Jupyter Lab 中打开并执行所有单元格。

---

## 📊 使用指南

### 1. 配置数据源

在 Notebook 的第一部分找到配置字典：

```python
CONFIG = {
    'SOURCE': 'MOCK',  # 切换为 'WIND_EDB' 使用真实数据
    'TICKER': 'M0017142',  # Wind EDB Ticker（地方债利差）
    'START_DATE': '2018-01-01',
    'END_DATE': '2025-12-31',
    # ... 其他参数
}
```

### 2. 执行分析流程

Notebook 分为四个主要部分：

#### **第一部分: 数据加载与清洗**
- 自动处理缺失值和异常值
- 输出数据质量报告

#### **第二部分: 模型锦标赛**

**模块 A - 波动率建模**:
- 拟合 GARCH(1,1)、EGARCH(1,1)、GJR-GARCH(1,1)
- 自动选出 AIC/BIC 最优模型
- 检测波动率不对称效应

**模块 B - 卡尔曼滤波**:
- 提取利差的"真实趋势"
- 识别均值回归交易机会

**模块 C - 极值理论**:
- 计算 99% EVT-VaR
- 量化尾部风险

#### **第三部分: 可视化仪表盘**

生成三张专业图表（全中文）：
1. **信号与趋势图** - 卡尔曼滤波 vs 原始利差 + 交易信号标记
2. **波动率结构图** - 条件波动率 + 危机模式识别
3. **尾部风险锥** - 收益率分布 + Student-t 拟合 + VaR 标记

#### **第四部分: 战略报告**

自动生成包含以下内容的分析报告：
- 模型选择结果与理由
- 波动率不对称效应检验
- 当前风险等级评估
- 具体交易建议（入场/目标/止损）

---

## 💡 核心技术亮点

### 1. 为什么用 "Model Tournament"？

金融市场的动态特征会随时间变化（Regime Switching）。单一模型可能在某些时期表现良好，但在其他时期失效。通过让多个模型竞争，我们可以：
- 避免模型选择偏差（Model Selection Bias）
- 自适应市场环境变化
- 提高预测稳健性

### 2. 为什么用 Student-t 分布而非正态分布？

金融收益率具有**肥尾特征**（Fat Tails）——极端值出现的频率远高于正态分布预测。Student-t 分布通过自由度参数 `df` 捕捉这一特征：
- `df` 越小，尾部越重
- 实际中国债券市场 `df` 通常在 3-7 之间

### 3. 为什么用 MAD 而非标准差？

Wind EDB 数据经常包含异常值（如 999、-999 占位符）。标准差对极端值非常敏感，会导致误判。MAD（Median Absolute Deviation）基于中位数，对离群点有抵抗力（Robust Estimator）。

### 4. 卡尔曼滤波器的实际意义？

市场报价 = 基本面价值 + 流动性溢价 + 微观结构噪音

我们真正关心的是第一项（基本面），但只能观测到总和。卡尔曼滤波器通过状态空间模型实现信号分离：
- **Smoothed State** → 基本面利差（用于制定战略）
- **Deviation** → 短期错误定价（用于交易）

---

## 📈 输出示例

### 模型锦标赛结果

```
================================================================
开始 GARCH 模型锦标赛
================================================================

[1/3] 拟合 GARCH(1,1)...
   AIC=4523.45, BIC=4537.21

[2/3] 拟合 EGARCH(1,1)...
   AIC=4512.33, BIC=4531.15
   非对称系数 γ = -0.1234 (负冲击放大波动)

[3/3] 拟合 GJR-GARCH(1,1)...
   AIC=4518.67, BIC=4537.89

================================================================
🏆 锦标赛获胜者: EGARCH
   AIC = 4512.33
   BIC = 4531.15
================================================================
```

### 战略建议输出

```
【四、行动建议】
────────────────────────────────────────────────────────────

  基于当前分析,建议:

  1. 方向性策略: 🔴 做空利差 (预期收窄)
     - 入场点: 125.43 bps
     - 目标价: 118.76 bps (回归趋势)
     - 止损点: 138.91 bps (当前+VaR)

  2. 风险管理:
     - 单日 VaR 限额: 13.48 bps
     - 建议仓位规模: 假设风险预算为 R bps,则最大名义敞口 = R / 13.48

  3. 关键监控指标:
     - 偏离度: 当前 +1.87σ → 警戒线 ±1.5σ, 止损线 ±2.5σ
     - 波动率: 当前 8.34 → 上升 20% 以上需重新评估风险敞口
     - 趋势: 若卡尔曼趋势突破 135.44 bps,说明市场regime切换
```

---

## 🔧 参数调优指南

### GARCH 模型参数

```python
CONFIG = {
    'GARCH_P': 1,  # ARCH 项阶数（通常 1 就够）
    'GARCH_Q': 1,  # GARCH 项阶数（通常 1 就够）
}
```

💡 **经验法则**: 除非有明确证据，否则坚持 (1,1)。更高阶数会过拟合。

### EVT 阈值选择

```python
CONFIG = {
    'EVT_THRESHOLD_PERCENTILE': 0.95,  # 95% 分位数
}
```

💡 **权衡**:
- **太低 (如 0.90)**: 引入非极端值，GPD 拟合效果差
- **太高 (如 0.99)**: 极值样本太少，估计不稳定
- **推荐**: 0.93 - 0.97 之间

### MAD 异常值阈值

```python
CONFIG = {
    'MAD_THRESHOLD': 5.0,  # MAD 倍数
}
```

💡 **建议**:
- **保守 (3.0)**: 更多异常值被剔除
- **宽松 (7.0)**: 保留更多数据
- **默认 (5.0)**: 平衡的选择

---

## 🎯 适用场景

本框架可直接应用于以下资产的利差分析：

✅ 地方政府债券利差（相对国债）
✅ 企业债信用利差（AA vs AAA）
✅ 银行间-交易所利差
✅ 期限利差（10Y - 2Y）
✅ 跨市场利差（中美国债）

**只需修改 `TICKER` 参数，无需改动代码逻辑。**

---

## ⚠️ 重要免责声明

1. **本框架为学术研究工具**，不构成投资建议
2. **所有模型都是对现实的简化**，实际交易需结合：
   - 宏观基本面分析
   - 政策环境研判
   - 市场微观结构
   - 流动性状况
3. **历史统计特征不保证未来延续**，特别是在：
   - 政策急转弯时期
   - 系统性危机爆发时
   - 市场结构性变化时

---

## 📚 参考文献

### GARCH 模型族
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity". *Journal of Econometrics*.
- Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New Approach". *Econometrica*.
- Glosten, L.R., Jagannathan, R., & Runkle, D.E. (1993). "On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks". *Journal of Finance*.

### 极值理论
- McNeil, A.J., & Frey, R. (2000). "Estimation of Tail-Related Risk Measures for Heteroscedastic Financial Time Series". *Journal of Empirical Finance*.
- Embrechts, P., Klüppelberg, C., & Mikosch, T. (1997). *Modelling Extremal Events for Insurance and Finance*. Springer.

### 卡尔曼滤波器
- Durbin, J., & Koopman, S.J. (2012). *Time Series Analysis by State Space Methods*. Oxford University Press.

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

如果您有以下改进建议，请联系：
- 新的模型变体（如 FIGARCH、Realized GARCH）
- 机器学习方法集成（如 LSTM 辅助预测）
- 高频数据适配
- 其他资产类别的应用案例

---

## 📄 License

MIT License - 详见 LICENSE 文件

---

## 👨‍💼 Author

**宏观对冲基金量化研究团队**

*"Let the data speak, and let the models compete."*
