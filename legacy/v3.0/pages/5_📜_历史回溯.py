"""
历史回溯页面 - 滚动统计与事件检测
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared_state import (
    init_page, render_sidebar, ensure_analysis,
    get_results, safe_metric, render_app_footer,
    calculate_rolling_stats, detect_historical_events,
    plot_rolling_stats, plot_percentile_chart,
    render_page_header
)

# ============================================================================
# 页面初始化
# ============================================================================

theme = init_page(page_title="历史回溯", page_icon="📜")
config, run_analysis = render_sidebar()
ensure_analysis(config, run_analysis)

render_page_header("📜 历史回溯", "Rolling Statistics & Event Detection")

# ============================================================================
# 历史回溯内容
# ============================================================================

results = get_results()
if results:
    clean_data = results['clean_data']

    # 核心说明
    st.info("""
    **📖 历史回溯**: 滚动统计与事件检测

    • **20天窗口**: 短期趋势，适合交易策略
    • **60天窗口**: 中期趋势，平衡噪音与滞后
    • **120天窗口**: 半年趋势，适合季度调仓
    • **252天窗口**: 年度趋势，反映长期结构性变化
    • **事件检测**: 识别历史异常波动，与政策变化相关
    """)

    st.divider()

    # 窗口选择解释
    window = st.selectbox("滚动窗口", [20, 60, 120, 252], index=1)

    rolling = calculate_rolling_stats(clean_data, window)
    st.plotly_chart(plot_rolling_stats(rolling, clean_data, theme), use_container_width=True)

    # 分位数分析
    st.markdown("### 📊 历史分位数分析")
    st.plotly_chart(plot_percentile_chart(clean_data, theme=theme), use_container_width=True)

    # 事件检测
    events = detect_historical_events(clean_data)
    if not events.empty:
        st.markdown("### 📌 历史重要事件")
        st.caption("基于统计方法检测的异常波动事件，可能与政策变化、市场冲击相关。")
        st.dataframe(events.head(10), use_container_width=True, hide_index=True)

render_app_footer()