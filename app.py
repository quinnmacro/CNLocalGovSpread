"""
中国地方债利差分析仪表板 - 首页
Streamlit 多页架构 v3.0

首页功能:
- 关键指标概览 (当前利差、历史均值、波动率、VaR、信号偏离)
- 风险状态总览
- 利差位置/波动率/风险评估快速解读
- 市场背景介绍

运行方式: streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from shared_state import (
    init_page, render_sidebar, ensure_analysis,
    get_results, safe_metric, render_app_footer,
    check_risk_alerts, get_risk_score,
    get_spread_position_comment, get_volatility_comment, get_var_comment,
    render_page_header
)
from market_status import MarketStatusGauge
from province_cluster import ProvinceClusterMap
import numpy as np

# ============================================================================
# 页面初始化
# ============================================================================

theme = init_page()
config, run_analysis = render_sidebar()

# 页面标题
render_page_header(
    "📊 中国地方政府债券利差分析",
    "Advanced Econometric Framework for China Local Government Bond Spread"
)

# 市场背景介绍
st.markdown("""
---
**📊 中国地方政府债券市场概览**

地方债利差 = 地方债收益率 - 国债收益率，反映市场对地方债的风险定价：
- **信用风险溢价**: 市场对地方政府偿债能力的差异化定价
- **流动性溢价**: 地方债交易活跃度低于国债，需额外补偿
- **政策预期**: 市场对财政政策、债务化解的预期变化

⚠️ **免责声明**: 本分析仅供学术研究和教育目的，不构成任何投资建议。所有模型都是对现实的简化，历史表现不代表未来收益。

---
""")

# ============================================================================
# 执行分析
# ============================================================================

ensure_analysis(config, run_analysis)

# ============================================================================
# 首页仪表板
# ============================================================================

results = get_results()
if results:
    clean_data = results['clean_data']
    returns = results['returns']
    deviation = results['deviation']
    winner = results['winner']
    winner_vol = results['winner_vol']
    evt = results['evt']
    var = results['var']
    es = results['es']

    current_spread = clean_data['spread'].iloc[-1]
    spread_mean = clean_data['spread'].mean()
    spread_std = clean_data['spread'].std()
    current_vol = winner_vol.iloc[-1]
    dev_val = deviation.iloc[-1]

    # =========================================================================
    # 关键指标概览
    # =========================================================================
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        safe_metric("当前利差", current_spread)
    with col2:
        safe_metric("历史均值", spread_mean)
    with col3:
        safe_metric("波动率", current_vol)
    with col4:
        safe_metric("99% VaR", var)
    with col5:
        if not np.isnan(dev_val):
            status = "🔴 高估" if dev_val > 1.5 else ("🟢 低估" if dev_val < -1.5 else "🟡 正常")
            st.metric("信号偏离", f"{dev_val:.2f}σ", help=status)

    # 风险评分一行
    alerts = check_risk_alerts(clean_data, returns, evt, results['vol_modeler'])
    risk_score = get_risk_score(alerts)
    st.markdown(f"**风险状态**: {risk_score['level']} | 🔴 {risk_score['danger_count']} | 🟡 {risk_score['warning_count']} | 🟢 {risk_score['total_alerts'] - risk_score['danger_count'] - risk_score['warning_count']}")

    # =========================================================================
    # 市场状态仪表 + 指标联动雷达图
    # =========================================================================
    gauge = MarketStatusGauge(
        clean_data, returns,
        smoothed=results['smoothed'],
        deviation=deviation,
        vol_modeler=results['vol_modeler'],
        evt=evt
    )
    gauge_status = gauge.get_market_status()

    st.markdown("#### 🎛️ 市场状态仪表")
    gauge_col, radar_col = st.columns([3, 2])
    with gauge_col:
        gauge_fig = gauge.plot_status_gauge(theme='light' if theme != 'dark' else 'dark')
        st.plotly_chart(gauge_fig, use_container_width=True)
    with radar_col:
        radar_fig = gauge.plot_indicator_linkage(theme='light' if theme != 'dark' else 'dark')
        st.plotly_chart(radar_fig, use_container_width=True)

    # 指标详情
    indicator_details = gauge._indicator_scores
    detail_cols = st.columns(5)
    indicator_names = {
        'spread_position': '利差定位',
        'volatility_regime': '波动率状态',
        'var_breach': 'VaR突破',
        'signal_deviation': '信号偏离',
        'trend_momentum': '趋势动量'
    }
    for i, (key, name) in enumerate(indicator_names.items()):
        with detail_cols[i]:
            score = indicator_details[key]['score']
            if score < 20:
                icon = '🟢'
            elif score < 40:
                icon = '🔵'
            elif score < 60:
                icon = '🟡'
            elif score < 80:
                icon = '🟠'
            else:
                icon = '🔴'
            st.metric(f"{icon} {name}", f"{score:.0f}")

    # =========================================================================
    # 省级利差聚类分析
    # =========================================================================
    st.divider()
    st.markdown("#### 🗺️ 省级利差聚类分析")
    pcm = ProvinceClusterMap(n_clusters=4)
    pcm.run_clustering()

    map_col, radar_col = st.columns([3, 2])
    with map_col:
        geo_fig = pcm.plot_choropleth_map(theme='light' if theme != 'dark' else 'dark')
        st.plotly_chart(geo_fig, use_container_width=True)
    with radar_col:
        cluster_radar = pcm.plot_cluster_comparison(theme='light' if theme != 'dark' else 'dark')
        st.plotly_chart(cluster_radar, use_container_width=True)

    # 聚类摘要
    cluster_stats = pcm.get_cluster_stats()
    summary_cols = st.columns(4)
    for i, (c, s) in enumerate(cluster_stats.items()):
        with summary_cols[i]:
            risk_icon = {'高风险': '🔴', '中等风险': '🟡', '低风险': '🔵', '极低风险': '🟢'}
            icon = risk_icon.get(s['risk_level'], '⚪')
            st.metric(
                f"{icon} 簇{c}",
                f"{s['n_provinces']}省 | {s['risk_level']}",
                f"均值 {s['mean_spread_avg']:.1f} bps"
            )

    # 热力图 (展开查看)
    with st.expander("🔍 查看省级聚类热力图"):
        heatmap_fig = pcm.plot_cluster_heatmap(theme='light' if theme != 'dark' else 'dark')
        st.plotly_chart(heatmap_fig, use_container_width=True)

    # =========================================================================
    # 快速解读卡片
    # =========================================================================
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📊 利差位置分析")
        st.info(get_spread_position_comment(current_spread, spread_mean, spread_std))

    with col2:
        st.markdown("#### 📈 波动率状态")
        mean_vol = winner_vol.mean()
        st.info(get_volatility_comment(current_vol, mean_vol))

    with col3:
        st.markdown("#### ⚠️ 风险评估")
        st.info(get_var_comment(var, es))

    # 导航提示
    st.divider()
    st.markdown("""
    **👉 使用左侧导航栏进入详细分析页面：**
    - 📈 信号分析 - 卡尔曼滤波趋势与交易信号
    - 📉 波动率分析 - GARCH模型锦标赛与状态切换
    - ⚠️ 风险分析 - 极值理论VaR/ES
    - 🎯 情景分析 - 压力测试与蒙特卡洛模拟
    - 📜 历史回溯 - 滚动统计与事件检测
    - 📋 报告中心 - 多格式报告生成
    """)

# ============================================================================
# 页脚
# ============================================================================
render_app_footer()