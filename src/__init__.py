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
from .ml_volatility import MLVolatilityModeler
from .calibration import ParameterCalibrator
from .market_status import MarketStatusGauge
from .province_cluster import ProvinceClusterMap
from .report_gen import ReportGenerator, TEMPLATES
from .export import export_to_excel
from .styles import (
    metric_card,
    alert_box,
    section_header,
    apply_theme,
    render_page_header,
    render_footer,
)
from .content import (
    get_spread_position_comment,
    get_volatility_comment,
    get_var_comment,
)
from .scenarios import (
    run_stress_test,
    run_multi_scenario_stress,
    run_monte_carlo,
    plot_mc_simulation,
    run_sensitivity_analysis,
    calculate_rolling_stats,
    detect_historical_events,
    plot_rolling_stats,
    plot_percentile_chart,
)
from .alerts import (
    check_risk_alerts,
    get_risk_score,
    generate_alert_history,
    plot_alert_timeline,
    plot_risk_gauge,
    plot_risk_summary,
    get_default_thresholds,
    validate_thresholds,
    format_alert_message,
    get_alert_summary,
)

__version__ = '3.0.0'
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
    'MLVolatilityModeler',
    'ParameterCalibrator',
    'MarketStatusGauge',
    'ProvinceClusterMap',
    'ReportGenerator',
    'TEMPLATES',
    'export_to_excel',
    'metric_card',
    'alert_box',
    'section_header',
    'apply_theme',
    'render_page_header',
    'render_footer',
    'get_spread_position_comment',
    'get_volatility_comment',
    'get_var_comment',
    'run_stress_test',
    'run_multi_scenario_stress',
    'run_monte_carlo',
    'plot_mc_simulation',
    'run_sensitivity_analysis',
    'calculate_rolling_stats',
    'detect_historical_events',
    'plot_rolling_stats',
    'plot_percentile_chart',
    'check_risk_alerts',
    'get_risk_score',
    'generate_alert_history',
    'plot_alert_timeline',
    'plot_risk_gauge',
    'plot_risk_summary',
    'get_default_thresholds',
    'validate_thresholds',
    'format_alert_message',
    'get_alert_summary',
]
