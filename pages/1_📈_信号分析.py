"""
信号分析页面 - 卡尔曼滤波趋势与交易信号
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared_state import (
    init_page, render_sidebar, ensure_analysis,
    get_results, safe_metric, render_app_footer,
    plot_signal_trend, render_metric_interpretation,
    render_trading_advice, alert_box, render_page_header
)
import numpy as np

# ============================================================================
# 页面初始化
# ============================================================================

theme = init_page(page_title="信号分析", page_icon="📈")
config, run_analysis = render_sidebar()
ensure_analysis(config, run_analysis)

render_page_header("📈 信号分析", "Kalman Filter Signal Extraction")

# ============================================================================
# 信号分析内容
# ============================================================================

results = get_results()
if results:
    clean_data = results['clean_data']
    deviation = results['deviation']
    smoothed = results['smoothed']
    winner_vol = results['winner_vol']
    dev_val = deviation.iloc[-1]

    # 核心说明
    st.info("""
    **📖 信号分析**: 卡尔曼滤波从市场噪音中提取真实趋势

    • **趋势线** = 卡尔曼平滑后的基本面利差（状态转移噪音 σ_η）
    • **原始线** = 市场观测利差（含短期流动性噪音）
    • **偏离度** = 标准化创新，衡量当前利差相对趋势的异常程度
    • **交易信号**: 偏离度 > 1.5σ 押注利差收敛（做空利差），< -1.5σ 押注利差扩大（做多利差）
    • **注意**: 信号基于均值回归假设，强趋势市场中可能失效
    """)

    st.divider()

    # 图表
    fig1 = plot_signal_trend(clean_data, smoothed, deviation, theme)
    st.plotly_chart(fig1, use_container_width=True)

    # 指标解读
    col1, col2, col3 = st.columns(3)
    with col1:
        safe_metric("趋势水平", smoothed.iloc[-1])
        render_metric_interpretation("spread")
    with col2:
        safe_metric("偏离度", dev_val, "σ")
        render_metric_interpretation("deviation")
    with col3:
        safe_metric("波动率", winner_vol.iloc[-1])
        render_metric_interpretation("volatility")

    # 交易信号解读
    st.markdown("### 🎯 交易信号解读")
    if dev_val > 1.5:
        alert_box(f"**做空信号**: 利差高估 {dev_val:.2f}σ", "danger")
        render_trading_advice("sell")
    elif dev_val < -1.5:
        alert_box(f"**做多信号**: 利差低估 {dev_val:.2f}σ", "success")
        render_trading_advice("buy")
    else:
        alert_box(f"**中性信号**: 偏离度 {dev_val:.2f}σ", "info")
        render_trading_advice("neutral")

render_app_footer()