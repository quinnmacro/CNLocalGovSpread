"""
中国地方债利差分析仪表板
Streamlit 简单版

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
from visualization import plot_signal_trend, plot_volatility_structure, plot_tail_risk
from export import export_to_excel

# 页面配置
st.set_page_config(
    page_title="中国地方债利差分析",
    page_icon="📊",
    layout="wide"
)

st.title("📊 中国地方政府债券利差分析")
st.markdown("**Advanced Econometric Framework for China Local Government Bond Spread**")
st.markdown("---")

# ============ 侧边栏配置 ============
with st.sidebar:
    st.header("⚙️ 参数配置")

    data_source = st.selectbox(
        "数据源",
        ["MOCK", "WIND_EDB"],
        index=0,
        help="MOCK为模拟数据，WIND_EDB需要配置Wind API"
    )

    st.subheader("日期范围")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "开始日期",
            value=pd.Timestamp("2018-01-01")
        )
    with col2:
        end_date = st.date_input(
            "结束日期",
            value=pd.Timestamp("2025-12-31")
        )

    st.subheader("风险参数")
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

    st.markdown("---")
    run_analysis = st.button("🚀 运行分析", type="primary", use_container_width=True)

# ============ 主分析逻辑 ============
if run_analysis:
    config = {
        'SOURCE': data_source,
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
    with st.spinner("加载数据..."):
        engine = DataEngine(config)
        engine.load_data()
        clean_data = engine.clean_data()
        returns = engine.get_returns()

    st.success(f"✓ 数据加载完成: {len(clean_data)} 个交易日")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 信号分析",
        "📉 波动率",
        "⚠️ 风险分析",
        "📋 战略报告"
    ])

    # Tab 1: 信号分析
    with tab1:
        st.subheader("卡尔曼滤波信号提取")

        with st.spinner("拟合卡尔曼滤波..."):
            kalman = KalmanSignalExtractor(clean_data['spread'])
            smoothed = kalman.fit()
            deviation = kalman.get_signal_deviation()

        fig1 = plot_signal_trend(clean_data, smoothed, deviation)
        st.plotly_chart(fig1, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("当前利差", f"{clean_data['spread'].iloc[-1]:.2f} bps")
        with col2:
            st.metric("趋势水平", f"{smoothed.iloc[-1]:.2f} bps")
        with col3:
            dev_val = deviation.iloc[-1]
            dev_status = "🔴 高估" if dev_val > 1.5 else ("🟢 低估" if dev_val < -1.5 else "🟡 正常")
            st.metric("偏离度", f"{dev_val:.2f} σ", dev_status)

    # Tab 2: 波动率分析
    with tab2:
        st.subheader("GARCH模型锦标赛")

        with st.spinner("运行GARCH锦标赛..."):
            vol_modeler = VolatilityModeler(returns)
            winner = vol_modeler.run_tournament()
            winner_vol = vol_modeler.get_conditional_volatility(winner)

        fig2 = plot_volatility_structure(winner_vol, winner)
        st.plotly_chart(fig2, use_container_width=True)

        # 状态切换
        st.subheader("波动率状态切换")
        with st.spinner("检测波动率状态..."):
            detector = RegimeDetector(winner_vol, n_regimes=3)
            detector.fit()
            current_regime = detector.get_current_regime()

        regime_names = {0: '🟢 低波动', 1: '🟡 中波动', 2: '🔴 高波动'}
        regime_colors = {0: 'green', 1: 'orange', 2: 'red'}

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("获胜模型", winner)
        with col2:
            st.metric("当前波动率", f"{winner_vol.iloc[-1]:.2f} bps")
        with col3:
            st.markdown(f"**当前状态**: {regime_names[current_regime]}")

        # 状态统计
        st.markdown("#### 状态统计")
        for i in range(3):
            stats = detector.regime_stats[i]
            st.write(f"{regime_names[i]}: 平均波动率 {stats['mean']:.2f} bps, 占比 {stats['pct']:.1f}%")

    # Tab 3: 风险分析
    with tab3:
        st.subheader("极值理论(EVT)风险分析")

        with st.spinner("拟合EVT模型..."):
            evt = EVTRiskAnalyzer(
                returns,
                threshold_percentile=evt_threshold,
                confidence=var_confidence
            )
            evt.fit_gpd()
            var = evt.calculate_var()
            es = evt.calculate_es()

        with st.spinner("计算Hill估计量..."):
            hill_index = evt.estimate_hill()

        fig3, evt_var_out, empirical_var = plot_tail_risk(returns, var, var_confidence)
        st.plotly_chart(fig3, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{var_confidence*100:.0f}% VaR", f"{var:.2f} bps", "单日最大风险")
        with col2:
            st.metric(f"{var_confidence*100:.0f}% ES", f"{es:.2f} bps", "尾部平均损失")
        with col3:
            tail_idx = evt.get_tail_index()
            if tail_idx:
                st.metric("尾部指数", f"{tail_idx:.2f}")

        # Hill与GPD对比
        if evt.hill_estimator and evt.gpd_params:
            st.markdown("#### Hill估计量 vs GPD")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"GPD形状参数 ξ: {evt.gpd_params['shape']:.4f}")
            with col2:
                st.write(f"Hill形状参数 ξ: {evt.hill_estimator['shape']:.4f}")

    # Tab 4: 战略报告
    with tab4:
        st.subheader("战略分析报告")

        # 模型结果
        st.markdown("#### 模型锦标赛结果")
        st.write(f"**获胜模型**: {winner}")
        st.write(f"**AIC**: {vol_modeler.ic_scores[winner]['AIC']:.2f}")
        st.write(f"**BIC**: {vol_modeler.ic_scores[winner]['BIC']:.2f}")

        # 当前状态
        st.markdown("#### 当前风险状况")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**当前利差**: {clean_data['spread'].iloc[-1]:.2f} bps")
            st.write(f"**趋势水平**: {smoothed.iloc[-1]:.2f} bps")
            st.write(f"**偏离度**: {deviation.iloc[-1]:.2f} σ")
        with col2:
            st.write(f"**当前波动率**: {winner_vol.iloc[-1]:.2f} bps")
            st.write(f"**波动率状态**: {regime_names[current_regime]}")
            st.write(f"**VaR限额**: {var:.2f} bps")
            st.write(f"**ES**: {es:.2f} bps")

        # 交易建议
        st.markdown("#### 交易建议")
        dev_val = deviation.iloc[-1]
        if dev_val > 1.5:
            st.warning("🔴 做空利差 (预期收窄)")
            st.write(f"- 入场点: {clean_data['spread'].iloc[-1]:.2f} bps")
            st.write(f"- 目标价: {smoothed.iloc[-1]:.2f} bps")
            st.write(f"- 止损点: {clean_data['spread'].iloc[-1] + var:.2f} bps")
        elif dev_val < -1.5:
            st.success("🟢 做多利差 (预期扩大)")
            st.write(f"- 入场点: {clean_data['spread'].iloc[-1]:.2f} bps")
            st.write(f"- 目标价: {smoothed.iloc[-1]:.2f} bps")
            st.write(f"- 止损点: {max(0, clean_data['spread'].iloc[-1] - var):.2f} bps")
        else:
            st.info("⚪ 中性观望 - 利差在合理区间")

        # 导出按钮
        st.markdown("---")
        if st.button("📥 导出Excel报告", type="primary"):
            output_path = "analysis_output.xlsx"
            export_to_excel(
                output_path,
                clean_data=clean_data,
                returns=returns,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=winner_vol,
                winner_model=winner,
                evt_var=var,
                evt_es=es,
                config=config
            )
            st.success(f"✓ 已导出到 {output_path}")

# ============ 页脚 ============
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <small>CNLocalGovSpread v1.2.0 | Author: Quinn Liu |
    <a href='https://github.com/quinnmacro/CNLocalGovSpread' target='_blank'>GitHub</a> |
    <a href='https://www.linkedin.com/in/liulu-math' target='_blank'>LinkedIn</a>
    </small>
</div>
""", unsafe_allow_html=True)
