# 中国地方政府债券利差高级计量经济学框架
## Advanced Econometric Framework for China Local Government Bond Spread Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/Tests-17%20passed-brightgreen.svg)

**Author: Quinn Liu**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/liulu-math)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/quinnmacro)

---

## 👨‍💼 About the Author

Quinn Liu is a senior fixed income investment professional with 8+ years of experience at global banks. Currently serving as Investment Manager at Bank of China Hong Kong Branch, managing a $50 billion USD foreign currency fixed income portfolio.

- **Expertise**: Fixed income portfolio management, quantitative strategy, derivatives pricing
- **Background**: M.S. in Financial Mathematics from Peking University
- **Interests**: Quantitative finance, AI/ML applications in finance, market microstructure

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

## 🏗️ 项目结构

```
CNLocalGovSpread/
├── src/                          # 源代码模块
│   ├── __init__.py
│   ├── data_engine.py            # 数据引擎
│   ├── volatility.py             # GARCH模型锦标赛
│   ├── kalman.py                 # 卡尔曼滤波器
│   ├── evt.py                    # 极值理论分析
│   ├── visualization.py          # 可视化函数
│   └── report.py                 # 战略报告生成
├── notebooks/
│   └── analysis.ipynb            # 主分析Notebook
├── tests/
│   └── test_all.py               # 单元测试
├── README.md
├── requirements.txt
└── LICENSE
```

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

### 运行测试

```bash
pytest tests/ -v
```

### 运行分析

```bash
cd notebooks
jupyter notebook analysis.ipynb
```

---

## 📊 技术架构

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

## 📄 License

MIT License - 详见 LICENSE 文件

---

*"Let the data speak, and let the models compete."*
