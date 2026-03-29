# CNLocalGovSpread 项目上下文

## 项目概述

中国地方政府债券利差高级计量经济学框架 - 用于求职展示的专业量化项目。

**作者**: Quinn Liu
**GitHub**: https://github.com/quinnmacro
**LinkedIn**: https://www.linkedin.com/in/liulu-math

## 技术栈

- **波动率建模**: GARCH/EGARCH/GJR-GARCH (arch库)
- **信号提取**: 卡尔曼滤波 (statsmodels)
- **风险分析**: 极值理论 EVT (scipy)
- **可视化**: Plotly 交互式图表

## 项目结构

```
src/
├── data_engine.py      # 数据加载与清洗
├── volatility.py       # GARCH模型锦标赛
├── kalman.py           # 卡尔曼滤波信号提取
├── evt.py              # 极值理论风险分析
├── visualization.py    # 可视化函数
└── report.py           # 战略报告生成

tests/
└── test_all.py         # 17个单元测试

notebooks/
└── analysis.ipynb      # 主分析notebook
```

## 运行命令

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest tests/ -v

# 运行分析
jupyter notebook china_localgov_bond_spread_analysis.ipynb
```

## 改进计划

详见 `.claude/improvement_plan.md`，包含:
- EWMA基准模型
- Expected Shortfall (CVaR)
- 波动率状态切换
- ML模型对比
- Streamlit仪表板

## 注意事项

- 数据源默认为MOCK模拟数据
- 生产环境需配置Wind EDB API
- 所有模型假设和局限性已在README中说明
