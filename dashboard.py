"""
中国地方债利差分析仪表板
Streamlit 现代化版本 v2.4 - 专业金融分析界面

功能模块:
1. 首页仪表板 - 关键指标与迷你图表
2. 信号分析 - 卡尔曼滤波信号提取
3. 波动率分析 - GARCH模型锦标赛
4. 风险分析 - 极值理论VaR/ES
5. 情景分析 - 压力测试与蒙特卡洛模拟
6. 历史回溯 - 滚动统计与事件检测
7. 报告中心 - 多格式报告生成

v2.4 更新:
- 每个Tab顶部添加直接可见的说明文字
- 市场背景默认展开
- 浅色模式改用专业灰蓝配色
- 默认深色主题

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
from content import (
    render_market_context,
    render_theory_expander,
    render_metric_interpretation,
    render_trading_advice,
    render_quick_reference,
    render_report_guide,
    get_spread_position_comment,
    get_volatility_comment,
    get_var_comment,
    KALMAN_THEORY,
    GARCH_THEORY,
    EVT_THEORY,
    SCENARIO_THEORY,
    HISTORY_THEORY,
    INTERPRETATION_GUIDE
)

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
# 页面配置 - 深色主题默认
# ============================================================================

st.set_page_config(
    page_title="中国地方债利差分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 主题切换 (默认深色)
theme = get_theme_toggle()
apply_theme(theme)

# 页面标题
render_page_header(
    "📊 中国地方政府债券利差分析",
    "Advanced Econometric Framework for China Local Government Bond Spread"
)

# 市场背景介绍 - 直接显示
st.markdown("""
---
**📊 中国地方政府债券市场概览**

地方债利差 = 地方债收益率 - 国债收益率，反映了：
- **信用风险溢价**: 市场对地方政府偿债能力的定价
- **流动性溢价**: 地方债交易活跃度低于国债
- **政策预期**: 对财政政策、债务化解的预期

---
""")

# ============================================================================
# 侧边栏配置
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ 配置")

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
        }.get(x, x)
    )

    st.divider()

    st.markdown("**📅 日期范围**")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始", value=pd.Timestamp("2018-01-01"), label_visibility="collapsed")
    with col2:
        end_date = st.date_input("结束", value=pd.Timestamp("2026-03-31"), label_visibility="collapsed")

    st.divider()

    st.markdown("**⚠️ 风险参数**")
    var_confidence = st.slider("VaR置信水平", 0.90, 0.99, 0.99, format="%.2f", label_visibility="collapsed")
    evt_threshold = st.slider("EVT阈值", 0.90, 0.99, 0.95, format="%.2f", label_visibility="collapsed")

    st.divider()
    run_analysis = st.button("🔄 重新分析", type="primary", use_container_width=True)

    if data_source == "CSV":
        if os.path.exists("data/local_gov_spread.csv"):
            st.success("✓ 数据就绪")
        else:
            st.error("✗ 数据文件不存在")

    # 分析框架快速参考
    st.divider()
    st.markdown("### 📚 分析框架")
    render_quick_reference()

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
        # 核心说明 - 直接显示
        st.info("""
        **📖 信号分析**: 卡尔曼滤波从市场噪音中提取真实趋势

        • **蓝色趋势线** = 基本面利差（卡尔曼平滑后）
        • **灰色原始线** = 市场观测利差（含噪音）
        • **偏离度** = (当前利差 - 趋势) / 噪音标准差
        • **交易信号**: 偏离度 > 1.5σ 做空，< -1.5σ 做多
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

    # =========================================================================
    # Tab 2: 波动率分析
    # =========================================================================
    with tab2:
        # 核心说明 - 直接显示
        st.info("""
        **📖 波动率分析**: GARCH模型锦标赛，让数据选择最优模型

        • **波动率聚集**: 平静期后往往平静，动荡期后往往动荡
        • **Student-t分布**: 捕捉厚尾特征（极端事件更频繁）
        • **模型选择**: AIC/BIC越小越好
        • **状态切换**: HMM识别低/中/高波动率状态
        """)

        st.divider()

        # 图表
        fig2 = plot_volatility_structure(winner_vol, winner, theme)
        st.plotly_chart(fig2, use_container_width=True)

        # 模型解读
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("获胜模型", winner)
            st.caption(f"基于AIC/BIC准则，{winner}模型对当前数据拟合最优。")
        with col2:
            safe_metric("当前波动率", current_vol)
            render_metric_interpretation("volatility")
        with col3:
            safe_metric("平均波动率", winner_vol.mean())
            st.caption("历史平均波动率，用于判断当前相对水平。")

        # 状态切换
        st.markdown("### 🔄 波动率状态分析")
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

            regime_desc = {
                0: "**低波动期**: 市场情绪稳定，流动性充裕。交易策略可适当积极。",
                1: "**中波动期**: 市场存在不确定性，需关注风险。建议控制仓位。",
                2: "**高波动期**: 危机模式，风险急剧上升。建议减仓或对冲。"
            }
            st.info(f"当前状态: **{regime_names[current_regime]}**\n\n{regime_desc[current_regime]}")
        except:
            pass

        # 模型对比表
        st.markdown("#### 📊 模型对比与选择")
        st.caption("信息准则越小越好。AIC倾向复杂模型，BIC倾向简约模型。")
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
        # 核心说明 - 直接显示
        st.warning("""
        **📖 风险分析**: 极值理论(EVT)专注于尾部风险

        • **VaR**: 给定置信度下的最大可能损失
        • **ES (预期损失)**: 超越VaR时的平均损失（更保守）
        • **尾部指数 ξ**: ξ > 0 表示厚尾分布
        • **为什么不用正态分布?** 严重低估极端损失概率
        """)

        st.divider()

        # 图表
        fig3, evt_var, empirical_var = plot_tail_risk(returns, var, var_confidence, theme)
        st.plotly_chart(fig3, use_container_width=True)

        # 风险指标解读
        st.markdown("### 📏 风险指标解读")
        col1, col2, col3 = st.columns(3)
        with col1:
            safe_metric(f"{var_confidence*100:.0f}% VaR", var)
            render_metric_interpretation("var_es")
        with col2:
            safe_metric(f"{var_confidence*100:.0f}% ES", es)
            render_metric_interpretation("var_es")
        with col3:
            tail_idx = evt.get_tail_index() if evt else None
            safe_metric("尾部指数 (ξ)", tail_idx)
            render_metric_interpretation("tail_index")

        # GPD参数解释
        if evt and evt.gpd_params:
            st.markdown("### 📐 广义帕累托分布参数")
            xi, sigma = evt.gpd_params['shape'], evt.gpd_params['scale']
            st.markdown(f"""
            **拟合结果**: ξ = {xi:.4f}, σ = {sigma:.4f}

            | 参数 | 估计值 | 解释 |
            |------|--------|------|
            | ξ (形状) | {xi:.4f} | {'厚尾，极端损失风险高' if xi > 0 else '薄尾'} |
            | σ (尺度) | {sigma:.4f} | 尾部衰减速度 |

            **结论**: {f'ξ = {xi:.4f} > 0，收益分布具有厚尾特征，极端损失发生概率高于正态分布预测。风险管理应采用EVT方法，而非传统正态假设。' if xi > 0 else '尾部特征不明显，可考虑简化模型。'}
            """)

    # =========================================================================
    # Tab 4: 情景分析
    # =========================================================================
    with tab4:
        # 核心说明 - 直接显示
        st.info("""
        **📖 情景分析**: 压力测试与蒙特卡洛模拟

        • **压力测试**: 假设特定利差冲击，计算VaR/ES变化
        • **蒙特卡洛**: 模拟大量可能路径，统计风险分布
        • **敏感性分析**: 评估参数变化对结果的影响
        • **注意**: 基于历史数据，黑天鹅事件可能未被包含
        """)

        st.divider()

        # 压力测试
        st.markdown("### 📉 压力测试")
        st.caption("假设特定利差冲击，评估极端情况下的风险敞口。")
        col1, col2 = st.columns(2)

        with col1:
            shock = st.slider("利差冲击 (bps)", -100, 100, 10, 5)
            stress = run_stress_test(returns, shock)
            st.metric("冲击后VaR", f"{stress['var']:.4f}")
            st.metric("冲击后ES", f"{stress['es']:.4f}")
            st.caption(f"利差{'扩大' if shock > 0 else '收窄'}{abs(shock)}bps时，预计最大损失。")

        with col2:
            st.markdown("#### 多情景压力测试")
            multi = run_multi_scenario_stress(returns)
            st.line_chart(pd.DataFrame(multi).set_index('shock')[['var', 'es']])
            st.caption("不同冲击幅度下的VaR/ES变化趋势。")

        # 蒙特卡洛
        st.divider()
        st.markdown("### 🎲 蒙特卡洛模拟")
        st.caption("基于历史分布特征，模拟未来多条可能路径，统计风险分布。")

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

            st.info(f"""
            **模拟结果解读**:
            - 模拟了{n_sim:,}条可能路径，预测{horizon}天后的利差变化
            - 99%概率下，损失不超过{mc['var_99']:.4f}
            - 如果发生极端情况，平均损失为{mc['es_99']:.4f}
            """)

    # =========================================================================
    # Tab 5: 历史回溯
    # =========================================================================
    with tab5:
        # 核心说明 - 直接显示
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

    # =========================================================================
    # Tab 6: 报告中心
    # =========================================================================
    with tab6:
        # 报告解读指南
        with st.expander("📋 报告解读指南", expanded=False):
            render_report_guide()

        st.divider()

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

        # 历史报告
        st.divider()
        st.markdown("### 📁 历史报告")
        history = get_report_history()
        if history is not None and not history.empty:
            st.dataframe(history, use_container_width=True, hide_index=True)
        else:
            st.info("暂无历史报告")

# ============================================================================
# 页脚
# ============================================================================
render_footer(
    version='2.4.0',
    author='Quinn Liu',
    github='https://github.com/quinnmacro/CNLocalGovSpread',
    linkedin='https://www.linkedin.com/in/liulu-math'
)
