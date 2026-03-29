"""
中国地方债利差分析仪表板
Streamlit 现代化版本 v2.0

功能模块:
1. 首页仪表板 - 关键指标与风险预警
2. 信号分析 - 卡尔曼滤波信号提取
3. 波动率分析 - GARCH模型锦标赛
4. 风险分析 - 极值理论VaR/ES
5. 情景分析 - 压力测试与蒙特卡洛模拟
6. 历史回溯 - 滚动统计与事件检测
7. 风险预警 - 实时预警系统
8. 报告中心 - 多格式报告生成

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
    plot_mc_paths,
    run_sensitivity_analysis,
    plot_sensitivity_analysis,
    calculate_rolling_stats,
    detect_historical_events,
    plot_rolling_stats,
    plot_percentile_chart
)
from alerts import (
    check_risk_alerts,
    get_risk_score,
    generate_alert_history,
    plot_risk_gauge,
    plot_risk_summary,
    plot_alert_timeline,
    get_alert_summary,
    format_alert_message
)
from report_gen import (
    ReportGenerator,
    generate_report,
    get_report_history,
    generate_quick_report
)
from export import export_to_excel

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

    # 多期限选择
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
        start_date = st.date_input(
            "开始日期",
            value=pd.Timestamp("2018-01-01")
        )
    with col2:
        end_date = st.date_input(
            "结束日期",
            value=pd.Timestamp("2026-03-31")
        )

    st.divider()

    st.subheader("⚠️ 风险参数")
    var_confidence = st.slider(
        "VaR置信水平",
        0.90, 0.99, 0.99,
        format="%.2f",
        help="置信水平越高，VaR越大"
    )

    evt_threshold = st.slider(
        "EVT阈值百分位",
        0.90, 0.99, 0.95,
        format="%.2f",
        help="极值理论阈值，推荐0.93-0.97"
    )

    st.divider()
    run_analysis = st.button("🚀 运行分析", type="primary", use_container_width=True)

    # 数据源状态
    if data_source == "CSV":
        csv_path = "data/local_gov_spread.csv"
        if os.path.exists(csv_path):
            st.success(f"✓ CSV数据已就绪")
        else:
            st.error("✗ CSV数据文件不存在")
    elif data_source == "WIND_EDB":
        st.warning("⚠️ 需要Wind终端连接")

# ============================================================================
# 主分析逻辑
# ============================================================================

# 初始化 session state
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

if run_analysis:
    # 创建配置
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
    with st.spinner("📥 加载数据..."):
        try:
            engine = DataEngine(config)
            engine.load_data()
            clean_data = engine.clean_data()
            returns = engine.get_returns()
            st.session_state.clean_data = clean_data
            st.session_state.returns = returns
            st.session_state.config = config
            st.success(f"✓ 数据加载完成: {len(clean_data)} 个交易日")
        except Exception as e:
            st.error(f"数据加载失败: {str(e)}")
            st.stop()

    # 卡尔曼滤波
    with st.spinner("拟合卡尔曼滤波..."):
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

    # 波动率建模
    with st.spinner("运行GARCH锦标赛..."):
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

    # EVT分析
    with st.spinner("拟合EVT模型..."):
        try:
            evt = EVTRiskAnalyzer(
                returns,
                threshold_percentile=evt_threshold,
                confidence=var_confidence
            )
            evt.fit_gpd()
            var = evt.calculate_var()
            es = evt.calculate_es()
            st.session_state.evt = evt
            st.session_state.var = var
            st.session_state.es = es
        except Exception as e:
            st.warning(f"EVT分析失败: {str(e)}")
            st.session_state.evt = None
            st.session_state.var = returns.quantile(var_confidence)
            st.session_state.es = returns.quantile(0.999)

    st.session_state.analysis_done = True

# 如果分析完成，显示仪表板
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

    # =========================================================================
    # 首页仪表板
    # =========================================================================
    section_header("📈 关键指标概览", "📊")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        safe_metric("当前利差", current_spread, "", "最新交易日利差值")
    with col2:
        safe_metric("历史均值", spread_mean, "", "全样本平均")
    with col3:
        current_vol = winner_vol.iloc[-1] if winner_vol is not None else None
        safe_metric("当前波动率", current_vol, "", f"{winner}模型")
    with col4:
        safe_metric(f"{var_confidence*100:.0f}% VaR", var, "", "单日最大风险")

    # 风险预警面板
    st.divider()
    section_header("⚠️ 风险预警", "🔔")

    alerts = check_risk_alerts(clean_data, returns, evt, vol_modeler)
    risk_score = get_risk_score(alerts)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(plot_risk_gauge(risk_score, theme), use_container_width=True)
    with col2:
        st.plotly_chart(plot_risk_summary(alerts, theme), use_container_width=True)

    # 预警详情
    with st.expander("📋 预警详情", expanded=True):
        for alert in alerts:
            alert_box(alert['message'], alert['level'])

    # =========================================================================
    # Tabs 导航
    # =========================================================================
    st.divider()
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 信号分析",
        "📉 波动率分析",
        "⚠️ 风险分析",
        "🎯 情景分析",
        "📜 历史回溯",
        "🔔 风险预警",
        "📋 报告中心"
    ])

    # =========================================================================
    # Tab 1: 信号分析
    # =========================================================================
    with tab1:
        section_header("卡尔曼滤波信号提取", "📈")

        # 信号图
        fig1 = plot_signal_trend(clean_data, smoothed, deviation, theme)
        st.plotly_chart(fig1, use_container_width=True)

        # 信号指标
        col1, col2, col3 = st.columns(3)
        with col1:
            safe_metric("当前利差", current_spread, "")
        with col2:
            safe_metric("趋势水平", smoothed.iloc[-1], "")
        with col3:
            dev_val = deviation.iloc[-1]
            if not np.isnan(dev_val) and not np.isinf(dev_val):
                if dev_val > 1.5:
                    dev_status = "🔴 高估"
                elif dev_val < -1.5:
                    dev_status = "🟢 低估"
                else:
                    dev_status = "🟡 正常"
                safe_metric("偏离度", dev_val, "σ", dev_status)
            else:
                safe_metric("偏离度", None, "σ", "计算异常")

        # 交易信号解读
        st.markdown("#### 信号解读")
        dev_val = deviation.iloc[-1]
        if not np.isnan(dev_val) and not np.isinf(dev_val):
            if dev_val > 1.5:
                alert_box(f"**做空信号**: 利差高估 {dev_val:.2f}σ，预期收窄", "danger")
            elif dev_val < -1.5:
                alert_box(f"**做多信号**: 利差低估 {dev_val:.2f}σ，预期扩大", "success")
            else:
                alert_box(f"**中性**: 利差在合理区间，偏离度 {dev_val:.2f}σ", "info")
        else:
            alert_box("数据平稳，无明显交易信号", "info")

    # =========================================================================
    # Tab 2: 波动率分析
    # =========================================================================
    with tab2:
        section_header("GARCH模型锦标赛", "📉")

        # 波动率图
        fig2 = plot_volatility_structure(winner_vol, winner, theme)
        st.plotly_chart(fig2, use_container_width=True)

        # 波动率指标
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("获胜模型", winner)
        with col2:
            current_vol = winner_vol.iloc[-1]
            if current_vol < 1e10 and not np.isnan(current_vol):
                safe_metric("当前波动率", current_vol, "")
            else:
                st.metric("当前波动率", "N/A")
        with col3:
            avg_vol = winner_vol.mean()
            if avg_vol < 1e10 and not np.isnan(avg_vol):
                safe_metric("平均波动率", avg_vol, "")
            else:
                st.metric("平均波动率", "N/A")

        # 状态切换
        st.subheader("波动率状态切换")
        try:
            with st.spinner("检测波动率状态..."):
                detector = RegimeDetector(winner_vol, n_regimes=3)
                detector.fit()
                current_regime = detector.get_current_regime()

            regime_names = {0: '🟢 低波动', 1: '🟡 中波动', 2: '🔴 高波动'}

            # 状态统计
            cols = st.columns(3)
            for i in range(3):
                with cols[i]:
                    stats = detector.regime_stats[i]
                    st.metric(
                        regime_names[i],
                        f"{stats['mean']:.4f}",
                        f"占比 {stats['pct']:.1f}%"
                    )

            st.info(f"当前状态: **{regime_names[current_regime]}**")
        except Exception as e:
            st.warning(f"状态检测不可用: {str(e)}")

    # =========================================================================
    # Tab 3: 风险分析
    # =========================================================================
    with tab3:
        section_header("极值理论(EVT)风险分析", "⚠️")

        # 尾部风险图
        try:
            fig3, evt_var_out, empirical_var = plot_tail_risk(returns, var, var_confidence, theme)
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.warning(f"图表生成失败: {str(e)}")

        # 风险指标
        col1, col2, col3 = st.columns(3)
        with col1:
            safe_metric(f"{var_confidence*100:.0f}% VaR", var, "", "单日最大风险")
        with col2:
            safe_metric(f"{var_confidence*100:.0f}% ES", es, "", "尾部平均损失")
        with col3:
            tail_idx = evt.get_tail_index() if evt else None
            if tail_idx and not np.isinf(tail_idx):
                safe_metric("尾部指数", tail_idx, "")
            else:
                st.metric("尾部指数", "N/A")

        # VaR 对比分析
        st.markdown("#### VaR 对比分析")
        print_var_comparison(evt_var_out, empirical_var)

    # =========================================================================
    # Tab 4: 情景分析
    # =========================================================================
    with tab4:
        section_header("情景分析", "🎯")

        # 压力测试
        with st.container(border=True):
            st.markdown("#### 📉 压力测试")
            col1, col2 = st.columns([1, 2])
            with col1:
                shock = st.slider("利差冲击", -100, 100, 10, 5, key="stress_shock")
                stress_results = run_stress_test(returns, shock)
            with col2:
                st.metric("压力VaR", f"{stress_results['var']:.4f}")
                st.metric("压力ES", f"{stress_results['es']:.4f}")
                st.metric("最大损失", f"{stress_results['max_loss']:.4f}")

        # 多情景压力测试
        with st.container(border=True):
            st.markdown("#### 📊 多情景压力测试")
            multi_stress = run_multi_scenario_stress(returns)
            stress_df = pd.DataFrame(multi_stress)
            st.line_chart(stress_df.set_index('shock')[['var', 'es']])

        # 蒙特卡洛模拟
        with st.container(border=True):
            st.markdown("#### 🎲 蒙特卡洛模拟")
            col1, col2 = st.columns([1, 2])
            with col1:
                n_sim = st.number_input("模拟次数", 1000, 100000, 10000, 1000, key="mc_nsim")
                horizon = st.number_input("预测天数", 1, 252, 10, key="mc_horizon")
                if st.button("运行模拟", key="run_mc"):
                    st.session_state['mc_results'] = run_monte_carlo(returns, n_sim, horizon)
            with col2:
                if 'mc_results' in st.session_state:
                    st.plotly_chart(plot_mc_simulation(st.session_state['mc_results'], theme), use_container_width=True)

        # 敏感性分析
        with st.container(border=True):
            st.markdown("#### 📈 敏感性分析")
            param = st.selectbox("参数", ["volatility", "mean", "df"], format_func=lambda x: {
                'volatility': '波动率',
                'mean': '均值',
                'df': '自由度'
            }.get(x, x), key="sens_param")
            sens_results = run_sensitivity_analysis(returns, param)
            st.plotly_chart(plot_sensitivity_analysis(sens_results, param, theme), use_container_width=True)

    # =========================================================================
    # Tab 5: 历史回溯
    # =========================================================================
    with tab5:
        section_header("历史回溯分析", "📜")

        # 滚动统计
        col1, col2 = st.columns([1, 2])
        with col1:
            window = st.selectbox("滚动窗口", [20, 60, 120, 252], index=1, key="rolling_window")
        rolling_stats = calculate_rolling_stats(clean_data, window)
        st.plotly_chart(plot_rolling_stats(rolling_stats, clean_data, theme), use_container_width=True)

        # 历史分位数
        st.plotly_chart(plot_percentile_chart(clean_data, theme=theme), use_container_width=True)

        # 历史事件
        st.markdown("#### 📌 历史重要事件")
        events = detect_historical_events(clean_data)
        if not events.empty:
            st.dataframe(events, use_container_width=True, hide_index=True)
        else:
            st.info("暂无检测到的重要事件")

    # =========================================================================
    # Tab 6: 风险预警
    # =========================================================================
    with tab6:
        section_header("风险预警系统", "🔔")

        # 预警阈值设置
        with st.expander("⚙️ 预警阈值设置"):
            col1, col2, col3 = st.columns(3)
            with col1:
                var_threshold = st.number_input("VaR预警阈值", 0.01, 0.10, 0.05, 0.01, key="alert_var")
            with col2:
                vol_percentile = st.slider("波动率预警百分位", 0.80, 0.99, 0.95, key="alert_vol")
            with col3:
                deviation_threshold = st.slider("偏离度预警阈值", 1.0, 3.0, 1.5, 0.1, key="alert_dev")

            # 重新检查预警
            if st.button("应用阈值", key="apply_thresholds"):
                alerts = check_risk_alerts(
                    clean_data, returns, evt, vol_modeler,
                    var_threshold, vol_percentile, deviation_threshold
                )
                st.session_state['custom_alerts'] = alerts

        # 当前预警状态
        st.markdown("#### 📊 当前预警状态")
        current_alerts = st.session_state.get('custom_alerts', alerts)
        for alert in current_alerts:
            alert_box(alert['message'], alert['level'])

        # 预警历史
        st.markdown("#### 📜 预警历史")
        alert_history = generate_alert_history(clean_data, returns)
        st.plotly_chart(plot_alert_timeline(alert_history, theme), use_container_width=True)

    # =========================================================================
    # Tab 7: 报告中心
    # =========================================================================
    with tab7:
        section_header("报告生成中心", "📋")

        # 报告配置
        with st.container(border=True):
            st.markdown("#### 📝 报告配置")
            col1, col2 = st.columns(2)
            with col1:
                report_title = st.text_input("报告标题", "地方债利差分析报告", key="report_title")
                report_format = st.selectbox("报告格式", ["HTML", "PDF", "Excel"], key="report_format")
            with col2:
                include_sections = st.multiselect(
                    "包含章节",
                    ["数据概览", "信号分析", "波动率分析", "风险分析", "交易建议"],
                    default=["数据概览", "信号分析", "风险分析"],
                    key="report_sections"
                )

        # 生成按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 生成报告", type="primary", key="gen_report"):
                with st.spinner("正在生成报告..."):
                    try:
                        report_path = generate_report(
                            clean_data, returns, kalman, vol_modeler, evt,
                            title=report_title,
                            format=report_format,
                            sections=include_sections
                        )
                        st.success(f"✓ 报告已生成: {report_path}")
                        st.session_state['last_report'] = report_path
                    except Exception as e:
                        st.error(f"报告生成失败: {str(e)}")

        with col2:
            if st.button("⚡ 快速报告", key="quick_report"):
                with st.spinner("正在生成快速报告..."):
                    try:
                        report_path = generate_quick_report(
                            clean_data, returns, kalman, vol_modeler, evt
                        )
                        st.success(f"✓ 快速报告已生成: {report_path}")
                        st.session_state['last_report'] = report_path
                    except Exception as e:
                        st.error(f"报告生成失败: {str(e)}")

        # 下载报告
        if 'last_report' in st.session_state and os.path.exists(st.session_state['last_report']):
            with open(st.session_state['last_report'], "rb") as f:
                st.download_button(
                    "📥 下载报告",
                    f,
                    file_name=os.path.basename(st.session_state['last_report']),
                    key="download_report"
                )

        # 历史报告
        with st.container(border=True):
            st.markdown("#### 📁 历史报告")
            history = get_report_history()
            if history is not None and not history.empty:
                st.dataframe(history, use_container_width=True, hide_index=True)
            else:
                st.info("暂无历史报告")

# ============================================================================
# 页脚
# ============================================================================
render_footer(
    version='2.0.0',
    author='Quinn Liu',
    github='https://github.com/quinnmacro/CNLocalGovSpread',
    linkedin='https://www.linkedin.com/in/liulu-math'
)
