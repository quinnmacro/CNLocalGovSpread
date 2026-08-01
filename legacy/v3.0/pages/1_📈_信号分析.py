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

    # 参数校准详情
    calibrated = results.get('calibrated') or {}
    kalman_window_cal = calibrated.get('kalman_window')
    signal_threshold_cal = calibrated.get('signal_threshold')
    kalman = results['kalman']

    if calibrated:
        with st.expander("⚙️ 参数校准详情 (v3.0)", expanded=False):
            st.markdown("""
            **参数自校准** 使用数据驱动方法估计最优参数，替代硬编码默认值。
            校准后的参数已自动应用于下游分析模块。
            """)
            cal_col1, cal_col2 = st.columns(2)
            with cal_col1:
                if kalman_window_cal is not None:
                    safe_metric("卡尔曼窗口", kalman_window_cal, "天",
                                help_text=f"数据驱动最优窗口 vs 默认60天")
                else:
                    st.metric("卡尔曼窗口", "60天 (默认)")
            with cal_col2:
                if signal_threshold_cal is not None:
                    safe_metric("信号阈值", signal_threshold_cal, "σ",
                                help_text=f"数据驱动偏离度阈值 vs 默认1.5σ")
                else:
                    st.metric("信号阈值", "1.5σ (默认)")

            # Kalman拟合状态
            kalman_status = "✅ Kalman主模型拟合成功" if kalman.success else "⚠️ 使用Fallback滚动均值"
            st.info(f"**拟合状态**: {kalman_status}")

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

    # 交易信号解读 (使用校准阈值或默认1.5σ)
    signal_threshold = signal_threshold_cal if signal_threshold_cal is not None else 1.5
    st.markdown("### 🎯 交易信号解读")
    threshold_source = "校准阈值" if signal_threshold_cal is not None else "默认阈值"
    st.caption(f"当前阈值: {signal_threshold:.2f}σ ({threshold_source})")

    if dev_val > signal_threshold:
        alert_box(f"**做空信号**: 利差高估 {dev_val:.2f}σ (> {signal_threshold:.2f}σ)", "danger")
        render_trading_advice("sell")
    elif dev_val < -signal_threshold:
        alert_box(f"**做多信号**: 利差低估 {dev_val:.2f}σ (< -{signal_threshold:.2f}σ)", "success")
        render_trading_advice("buy")
    else:
        alert_box(f"**中性信号**: 偏离度 {dev_val:.2f}σ (阈值 ±{signal_threshold:.2f}σ)", "info")
        render_trading_advice("neutral")

render_app_footer()