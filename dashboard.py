"""
中国地方债利差分析仪表板
Streamlit 现代化版本 v2.1

功能模块:
1. 首页仪表板 - 关键指标与迷你图表
2. 信号分析 - 卡尔曼滤波信号提取
3. 波动率分析 - GARCH模型锦标赛
4. 风险分析 - 极值理论VaR/ES
5. 情景分析 - 压力测试与蒙特卡洛模拟
6. 历史回溯 - 滚动统计与事件检测
7. 报告中心 - 多格式报告生成

运行方式: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_engine import DataEngine
from volatility import VolatilityModeler, RegimeDetector
from kalman import KalmanSignalExtractor
from evt import EVTRiskAnalyzer
from visualization import (
    plot_signal_trend,
    plot_volatility_structure,
    plot_tail_risk,
    print_var_comparison,
    plot_multi_tenor_spread,
    plot_tenor_spread_correlation,
    plot_tenor_spread_statistics
)
from styles import (
    apply_theme,
    get_theme_toggle,
    metric_card,
    alert_box,
    section_header,
    render_page_header,
    render_footer
)
from scenarios import (
    run_stress_test,
    run_multi_scenario_stress,
    run_monte_carlo,
    plot_mc_simulation,
    run_sensitivity_analysis,
    plot_sensitivity_analysis,
    calculate_rolling_stats,
    detect_historical_events,
    plot_rolling_stats,
    plot_percentile_chart
)
from alerts import check_risk_alerts, get_risk_score
from report_gen import generate_report, get_report_history, generate_quick_report

# ============================================================================
# 工具函数
# ============================================================================

def safe_format(value, unit="", decimals=4):
    """安全格式化数值，处理 inf/nan/极端值"""
    if value is None:
        return "N/A"
    try:
        if np.isnan(value) or np.isinf(value):
            return "N/A"
        if abs(value) > 1e10:
            return "N/A"
        if unit:
            return f"{value:.{decimals}f} {unit}"
        return f"{value:.{decimals}f}"
    except:
        return "N/A"


def safe_metric(label, value, unit="", help_text=None, delta=None):
    """安全显示 metric，处理异常值"""
    formatted = safe_format(value, unit)
    if delta is not None:
        st.metric(label, formatted, delta=delta, help=help_text)
    else:
        st.metric(label, formatted, help=help_text)


# ============================================================================
# 页面配置
# ============================================================================

st.set_page_config(
    page_title="中国地方债利差分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 主题切换
theme = get_theme_toggle()
apply_theme(theme)

# 页面标题
render_page_header(
    "📊 中国地方政府债券利差分析",
    "Advanced Econometric Framework for China Local Government Bond Spread"
)

# ============================================================================
# 侧边栏配置
# ============================================================================

with st.sidebar:
    st.header("⚙️ 参数配置")

    data_source = st.selectbox(
        "数据源",
        ["CSV", "MOCK", "WIND_EDB"],
        index=0,
        help="CSV为离线数据，MOCK为模拟数据，WIND_EDB需要配置Wind API"
    )

    spread_column = st.selectbox(
        "利差期限",
        ["spread_all", "spread_5y", "spread_10y", "spread_30y"],
        index=0,
        format_func=lambda x: {
            'spread_all': '综合利差',
            'spread_5y': '5年期',
            'spread_10y': '10年期',
            'spread_30y': '30年期'
        }.get(x, x),
        help="选择不同期限的利差数据"
    )

    st.divider()

    st.subheader("📅 日期范围")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", value=pd.Timestamp("2018-01-01"))
    with col2:
        end_date = st.date_input("结束日期", value=pd.Timestamp("2026-03-31"))

    st.divider()

    st.subheader("⚠️ 风险参数")
    var_confidence = st.slider("VaR置信水平", 0.90, 0.99, 0.99, format="%.2f")
    evt_threshold = st.slider("EVT阈值百分位", 0.90, 0.99, 0.95, format="%.2f")

    st.divider()
    run_analysis = st.button("🚀 运行分析", type="primary", use_container_width=True)

    if data_source == "CSV":
        if os.path.exists("data/local_gov_spread.csv"):
            st.success("✓ CSV数据已就绪")
        else:
            st.error("✗ CSV数据文件不存在")
    elif data_source == "WIND_EDB":
        st.warning("⚠️ 需要Wind终端连接")

# ============================================================================
# 主分析逻辑 - 自动预加载
# ============================================================================

# 首次访问自动加载
need_load = 'analysis_done' not in st.session_state or not st.session_state.analysis_done

if need_load:
    # 预加载遮罩
    load_placeholder = st.empty()

if run_analysis or need_load:
    if need_load:
        load_placeholder = st.status("⚡ 正在初始化分析引擎...", expanded=True)

    config = {
        'SOURCE': data_source,
        'CSV_PATH': 'data/local_gov_spread.csv',
        'SPREAD_COLUMN': spread_column,
        'TICKER': 'M0017142',
        'START_DATE': str(start_date),
        'END_DATE': str(end_date),
        'MAD_THRESHOLD': 5.0,
        'GARCH_P': 1,
        'GARCH_Q': 1,
        'VaR_CONFIDENCE': var_confidence,
        'EVT_THRESHOLD_PERCENTILE': evt_threshold
    }

    # 数据加载
    if need_load:
        load_placeholder.update(label="📥 加载数据...")
    try:
        engine = DataEngine(config)
        engine.load_data()
        clean_data = engine.clean_data()
        returns = engine.get_returns()
        st.session_state.clean_data = clean_data
        st.session_state.returns = returns
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        st.stop()

    # 卡尔曼滤波
    if need_load:
        load_placeholder.update(label="📈 拟合卡尔曼滤波...")
    try:
        kalman = KalmanSignalExtractor(clean_data['spread'])
        smoothed = kalman.fit()
        deviation = kalman.get_signal_deviation()
        st.session_state.kalman = kalman
        st.session_state.smoothed = smoothed
        st.session_state.deviation = deviation
    except Exception as e:
        st.error(f"卡尔曼滤波拟合失败: {str(e)}")
        st.stop()

    # GARCH模型
    if need_load:
        load_placeholder.update(label="📉 运行GARCH锦标赛...")
    try:
        vol_modeler = VolatilityModeler(returns)
        winner = vol_modeler.run_tournament()
        winner_vol = vol_modeler.get_conditional_volatility(winner)
        st.session_state.vol_modeler = vol_modeler
        st.session_state.winner = winner
        st.session_state.winner_vol = winner_vol
    except Exception as e:
        st.error(f"波动率建模失败: {str(e)}")
        st.stop()

    # EVT风险分析
    if need_load:
        load_placeholder.update(label="⚠️ 拟合EVT模型...")
    try:
        evt = EVTRiskAnalyzer(returns, threshold_percentile=evt_threshold, confidence=var_confidence)
        evt.fit_gpd()
        var = evt.calculate_var()
        es = evt.calculate_es()
        st.session_state.evt = evt
        st.session_state.var = var
        st.session_state.es = es
    except Exception as e:
        st.session_state.evt = None
        st.session_state.var = returns.quantile(var_confidence)
        st.session_state.es = returns.quantile(0.999)

    st.session_state.analysis_done = True

    if need_load:
        load_placeholder.update(label="✅ 分析完成!", state="complete")
        st.balloons()

# ============================================================================
# 分析结果展示
# ============================================================================

if st.session_state.analysis_done:
    clean_data = st.session_state.clean_data
    returns = st.session_state.returns
    kalman = st.session_state.kalman
    smoothed = st.session_state.smoothed
    deviation = st.session_state.deviation
    vol_modeler = st.session_state.vol_modeler
    winner = st.session_state.winner
    winner_vol = st.session_state.winner_vol
    evt = st.session_state.evt
    var = st.session_state.var
    es = st.session_state.es

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
        safe_metric(f"99% VaR", var)
    with col5:
        if not np.isnan(dev_val):
            status = "🔴 高估" if dev_val > 1.5 else ("🟢 低估" if dev_val < -1.5 else "🟡 正常")
            st.metric("信号偏离", f"{dev_val:.2f}σ", help=status)

    # 风险评分一行
    alerts = check_risk_alerts(clean_data, returns, evt, vol_modeler)
    risk_score = get_risk_score(alerts)
    st.markdown(f"**风险状态**: {risk_score['level']} | 🔴 {risk_score['danger_count']} | 🟡 {risk_score['warning_count']} | 🟢 {risk_score['total_alerts'] - risk_score['danger_count'] - risk_score['warning_count']}")

    # =========================================================================
    # Tabs 导航
    # =========================================================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 信号分析",
        "📉 波动率分析",
        "⚠️ 风险分析",
        "🎯 情景分析",
        "📜 历史回溯",
        "📋 报告中心"
    ])

    # =========================================================================
    # Tab 1: 信号分析
    # =========================================================================
    with tab1:
        fig1 = plot_signal_trend(clean_data, smoothed, deviation, theme)
        st.plotly_chart(fig1, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            safe_metric("趋势水平", smoothed.iloc[-1])
        with col2:
            safe_metric("偏离度", dev_val, "σ")
        with col3:
            safe_metric("波动率", winner_vol.iloc[-1])

        # 交易信号
        if dev_val > 1.5:
            alert_box(f"做空信号: 利差高估 {dev_val:.2f}σ", "danger")
        elif dev_val < -1.5:
            alert_box(f"做多信号: 利差低估 {dev_val:.2f}σ", "success")
        else:
            alert_box(f"中性: 偏离度 {dev_val:.2f}σ", "info")

    # =========================================================================
    # Tab 2: 波动率分析
    # =========================================================================
    with tab2:
        fig2 = plot_volatility_structure(winner_vol, winner, theme)
        st.plotly_chart(fig2, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("获胜模型", winner)
        with col2:
            safe_metric("当前波动率", current_vol)
        with col3:
            safe_metric("平均波动率", winner_vol.mean())

        # 状态切换
        try:
            detector = RegimeDetector(winner_vol, n_regimes=3)
            detector.fit()
            current_regime = detector.get_current_regime()
            regime_names = {0: '🟢 低波动', 1: '🟡 中波动', 2: '🔴 高波动'}

            cols = st.columns(3)
            for i in range(3):
                with cols[i]:
                    stats = detector.regime_stats[i]
                    st.metric(regime_names[i], f"{stats['mean']:.4f}", f"{stats['pct']:.1f}%")

            st.info(f"当前状态: **{regime_names[current_regime]}**")
        except:
            pass

        # 模型对比表
        st.markdown("#### 模型对比")
        model_data = []
        for m in vol_modeler.models.keys():
            model_data.append({
                '模型': m,
                'AIC': f"{vol_modeler.ic_scores[m]['AIC']:.1f}",
                'BIC': f"{vol_modeler.ic_scores[m]['BIC']:.1f}",
                '获胜': '✓' if m == winner else ''
            })
        st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)

    # =========================================================================
    # Tab 3: 风险分析
    # =========================================================================
    with tab3:
        fig3, evt_var, empirical_var = plot_tail_risk(returns, var, var_confidence, theme)
        st.plotly_chart(fig3, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            safe_metric(f"{var_confidence*100:.0f}% VaR", var)
        with col2:
            safe_metric(f"{var_confidence*100:.0f}% ES", es)
        with col3:
            tail_idx = evt.get_tail_index() if evt else None
            safe_metric("尾部指数", tail_idx)

        if evt and evt.gpd_params:
            st.markdown(f"**GPD参数**: ξ = {evt.gpd_params['shape']:.4f}, σ = {evt.gpd_params['scale']:.4f}")
            if evt.gpd_params['shape'] > 0:
                st.info("ξ > 0 表示厚尾分布，极端损失风险较高")

    # =========================================================================
    # Tab 4: 情景分析
    # =========================================================================
    with tab4:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 压力测试")
            shock = st.slider("利差冲击", -100, 100, 10, 5)
            stress = run_stress_test(returns, shock)
            st.metric("冲击后VaR", f"{stress['var']:.4f}")
            st.metric("冲击后ES", f"{stress['es']:.4f}")

        with col2:
            st.markdown("#### 多情景压力测试")
            multi = run_multi_scenario_stress(returns)
            st.line_chart(pd.DataFrame(multi).set_index('shock')[['var', 'es']])

        # 蒙特卡洛
        st.divider()
        st.markdown("#### 蒙特卡洛模拟")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            n_sim = st.number_input("模拟次数", 1000, 100000, 10000, 1000)
        with col2:
            horizon = st.number_input("预测天数", 1, 252, 10)
        with col3:
            if st.button("运行模拟", type="primary"):
                st.session_state['mc'] = run_monte_carlo(returns, n_sim, horizon)

        if 'mc' in st.session_state:
            mc = st.session_state['mc']
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("均值", f"{mc['mean']:.4f}")
            c2.metric("标准差", f"{mc['std']:.4f}")
            c3.metric("99% VaR", f"{mc['var_99']:.4f}")
            c4.metric("99% ES", f"{mc['es_99']:.4f}")
            st.plotly_chart(plot_mc_simulation(mc, theme), use_container_width=True)

    # =========================================================================
    # Tab 5: 历史回溯
    # =========================================================================
    with tab5:
        window = st.selectbox("滚动窗口", [20, 60, 120, 252], index=1)
        rolling = calculate_rolling_stats(clean_data, window)
        st.plotly_chart(plot_rolling_stats(rolling, clean_data, theme), use_container_width=True)
        st.plotly_chart(plot_percentile_chart(clean_data, theme=theme), use_container_width=True)

        events = detect_historical_events(clean_data)
        if not events.empty:
            st.markdown("#### 历史重要事件")
            st.dataframe(events.head(10), use_container_width=True, hide_index=True)

    # =========================================================================
    # Tab 6: 报告中心
    # =========================================================================
    with tab6:
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("报告标题", "地方债利差分析报告")
            fmt = st.selectbox("格式", ["HTML", "PDF", "Excel"])
        with col2:
            sections = st.multiselect(
                "章节",
                ["数据概览", "信号分析", "波动率分析", "风险分析", "交易建议"],
                default=["数据概览", "信号分析", "风险分析"]
            )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 生成报告", type="primary"):
                with st.spinner("生成中..."):
                    path = generate_report(clean_data, returns, kalman, vol_modeler, evt,
                                          title=title, format=fmt, sections=sections)
                    st.success(f"✓ {path}")
                    st.session_state['report'] = path

        with col2:
            if st.button("⚡ 快速报告"):
                path = generate_quick_report(clean_data, returns, kalman, vol_modeler, evt)
                st.success(f"✓ {path}")
                st.session_state['report'] = path

        if 'report' in st.session_state and os.path.exists(st.session_state['report']):
            with open(st.session_state['report'], "rb") as f:
                st.download_button("📥 下载", f, file_name=os.path.basename(st.session_state['report']), type="primary")

        history = get_report_history()
        if history is not None and not history.empty:
            st.dataframe(history, use_container_width=True, hide_index=True)

# ============================================================================
# 页脚
# ============================================================================
render_footer(
    version='2.1.0',
    author='Quinn Liu',
    github='https://github.com/quinnmacro/CNLocalGovSpread',
    linkedin='https://www.linkedin.com/in/liulu-math'
)
