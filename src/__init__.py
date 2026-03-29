"""
中国地方政府债券利差高级计量经济学框架
Advanced Econometric Framework for China Local Government Bond Spread Analysis

作者: Quinn Liu
GitHub: https://github.com/quinnmacro
LinkedIn: https://www.linkedin.com/in/liulu-math

核心理念: 模型锦标赛 (Model Tournament) - 不依赖单一模型，让数据选择最优动态特征

三大分析模块:
1. 波动率建模锦标赛 (GARCH/EGARCH/GJR-GARCH) - 捕捉不对称效应
2. 卡尔曼滤波器 - 从噪音中提取真实信号
3. 极值理论 (EVT) - 尾部风险量化
"""

from .data_engine import DataEngine
from .volatility import VolatilityModeler, RegimeDetector
from .kalman import KalmanSignalExtractor
from .evt import EVTRiskAnalyzer
from .visualization import (
    plot_signal_trend,
    plot_volatility_structure,
    plot_tail_risk,
    print_var_comparison,
    plot_multi_tenor_spread,
    plot_tenor_spread_correlation,
    plot_tenor_spread_statistics,
    plot_credit_spread_comparison,
    plot_spread_premium_analysis
)
from .report import generate_strategic_report
from .export import export_to_excel

__version__ = '2.1.0'
__author__ = 'Quinn Liu'

__all__ = [
    'DataEngine',
    'VolatilityModeler',
    'RegimeDetector',
    'KalmanSignalExtractor',
    'EVTRiskAnalyzer',
    'plot_signal_trend',
    'plot_volatility_structure',
    'plot_tail_risk',
    'print_var_comparison',
    'plot_multi_tenor_spread',
    'plot_tenor_spread_correlation',
    'plot_tenor_spread_statistics',
    'plot_credit_spread_comparison',
    'plot_spread_premium_analysis',
    'generate_strategic_report',
    'export_to_excel'
]
