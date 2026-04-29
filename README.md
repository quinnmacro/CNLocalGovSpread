# 中国地方政府债券利差高级计量经济学框架
## Advanced Econometric Framework for China Local Government Bond Spread Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Version](https://img.shields.io/badge/Version-3.0.0-brightgreen.svg)

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
1. **波动率建模锦标赛** - GARCH/EGARCH/GJR-GARCH + FIGARCH 长记忆检测 模型竞争
2. **卡尔曼滤波器** - 从市场噪音中提取真实信号
3. **极值理论 (EVT)** - 基于 GPD 的尾部风险量化

✅ **ML模型扩展** (v3.0新增):
- Random Forest / XGBoost / LSTM 波动率预测
- ML与GARCH统一评估 (AIC/BIC + RMSE/MAE)
- 特征工程 (9维滚动窗口统计量)

✅ **参数自校准** (v3.0新增):
- EWMA lambda (QLIKE优化)
- t分布 df (MLE估计)
- AR(1) phi (OLS回归)
- EVT阈值 (MEF稳定性检测)
- 卡尔曼窗口/信号阈值 (数据驱动)

✅ **交互式Dashboard 2.0** (v3.0重构):
- 多页面Streamlit架构 (app.py + pages/)
- 市场状态仪表盘 (5指标加权融合)
- 省份聚类地图 (31省层次聚类 + 地理分布)
- 深色/浅色双主题切换
- 情景分析（压力测试、蒙特卡洛模拟）
- 风险预警系统
- 报告生成中心（PDF/Excel/HTML/PPT + 3模板风格）
- 计量经济学教育内容

✅ **交互式可视化** - 使用 Plotly 生成专业图表（中文标注）

✅ **战略输出报告** - 自动生成可执行的交易建议和风险预警

---

## 🏗️ 项目结构

```
CNLocalGovSpread/
├── app.py                        # 多页面Dashboard入口 (v3.0)
├── shared_state.py               # Dashboard共享状态模块 (v3.0)
├── dashboard.py                  # 单页面Dashboard (向后兼容)
├── pages/                        # 多页面子页面 (v3.0)
│   ├── 1_📈_信号分析.py
│   ├── 2_📉_波动率分析.py
│   ├── 3_⚠️_风险分析.py
│   ├── 4_🎯_情景分析.py
│   ├── 5_📜_历史回溯.py
│   └── 6_📋_报告中心.py
├── src/                          # 源代码模块
│   ├── __init__.py               # v3.0.0
│   ├── data_engine.py            # 数据引擎
│   ├── volatility.py             # GARCH锦标赛 + FIGARCH (v3.0)
│   ├── kalman.py                 # 卡尔曼滤波器
│   ├── evt.py                    # 极值理论分析 (含MEF阈值选择 v3.0)
│   ├── visualization.py          # 可视化函数
│   ├── ml_volatility.py          # ML波动率模型 (v3.0)
│   ├── calibration.py            # 参数自校准 (v3.0)
│   ├── market_status.py          # 市场状态仪表 (v3.0)
│   ├── province_cluster.py       # 省份聚类地图 (v3.0)
│   ├── styles.py                 # 主题样式系统
│   ├── scenarios.py              # 情景分析模块
│   ├── alerts.py                 # 风险预警系统
│   ├── report_gen.py             # 报告生成中心
│   ├── content.py                # 计量经济学教育内容
│   ├── export.py                 # 数据导出
│   └── report.py                 # 战略报告生成
├── notebooks/
│   └── analysis.ipynb            # 主分析Notebook
├── tests/
│   ├── conftest.py               # 共享pytest fixtures + markers
│   ├── test_all.py               # 核心模块测试 (17)
│   ├── test_ml_volatility.py     # ML波动率测试 (15)
│   ├── test_calibration.py       # 参数校准测试 (39)
│   ├── test_figarch.py           # FIGARCH测试 (23)
│   ├── test_market_status.py     # 市场状态测试 (37)
│   ├── test_province_cluster.py  # 省份聚类测试 (43)
│   ├── test_alerts.py            # 预警系统测试 (37)
│   ├── test_scenarios.py         # 情景分析测试 (56)
│   ├── test_visualization.py     # 可视化测试 (80)
│   ├── test_report_gen.py        # 报告生成测试 (46)
│   ├── test_content.py           # 内容辅助函数测试 (26)
│   ├── test_styles.py            # 主题样式测试 (30)
│   ├── test_export.py            # 数据导出测试 (16)
│   ├── test_dashboard_integration.py # Dashboard集成测试 (55)
│   └── run_tests.sh              # CI友好测试运行器
├── CHANGELOG.md                  # 版本变更记录 (v3.0)
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
# 全部测试 (551个)
pytest tests/ -v

# CI友好运行器 (6种模式)
bash tests/run_tests.sh all      # 全部测试
bash tests/run_tests.sh quick    # 快速测试
bash tests/run_tests.sh count    # 仅计数
```

### 运行分析

```bash
cd notebooks
jupyter notebook analysis.ipynb
```

### 运行Dashboard (v3.0)

```bash
# 多页面Dashboard (推荐)
streamlit run app.py

# 单页面Dashboard (向后兼容)
streamlit run dashboard.py
```

---

## 📊 技术架构

### 模型实现细节

| 模块 | 技术 | 核心库 | 用途 |
|------|------|--------|------|
| **波动率建模** | GARCH/EGARCH/GJR-GARCH | `arch` | 捕捉波动率聚集和不对称效应 |
| **长记忆检测** | FIGARCH (GPH估计) | `numpy` | 检测波动率长记忆特征 |
| **ML波动率** | RF/XGBoost/LSTM | `sklearn/xgboost` | 机器学习波动率预测对比 |
| **参数校准** | QLIKE/MLE/OLS | `scipy/statsmodels` | 数据驱动参数估计 |
| **信号提取** | 卡尔曼滤波器 | `statsmodels` | 分离基本面信号与市场噪音 |
| **尾部风险** | 极值理论 (POT) | `scipy.stats` | 估计极端损失的概率分布 |
| **市场状态** | 5指标加权融合 | `numpy` | 实时市场健康度评估 |
| **报告生成** | PDF/Excel/HTML/PPT | `reportlab/python-pptx` | 多格式战略报告输出 + 3模板风格 |
| **省份聚类** | 层次聚类 (Ward) | `scipy` | 31省利差特征分组 |
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
