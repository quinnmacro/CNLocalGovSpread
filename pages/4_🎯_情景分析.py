"""
情景分析页面 - 压力测试与蒙特卡洛模拟
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared_state import (
    init_page, render_sidebar, ensure_analysis,
    get_results, safe_metric, render_app_footer,
    run_stress_test, run_multi_scenario_stress,
    run_monte_carlo, plot_mc_simulation, plot_mc_paths,
    run_sensitivity_analysis, plot_sensitivity_analysis,
    render_page_header
)
import pandas as pd

# ============================================================================
# 页面初始化
# ============================================================================

theme = init_page(page_title="情景分析", page_icon="🎯")
config, run_analysis = render_sidebar()
ensure_analysis(config, run_analysis)

render_page_header("🎯 情景分析", "Stress Testing & Monte Carlo Simulation")

# ============================================================================
# 情景分析内容
# ============================================================================

results = get_results()
if results:
    returns = results['returns']

    # 核心说明
    st.info("""
    **📖 情景分析**: 压力测试与蒙特卡洛模拟

    • **压力测试**: 假设特定利差冲击，评估极端情况下的VaR/ES变化（取右尾，关注利差扩大风险）
    • **蒙特卡洛**: AR(1)均值回归模型 + t分布残差，模拟未来路径，避免路径发散
    • **敏感性分析**: 评估波动率、均值、自由度等参数变化对风险指标的影响
    • **局限性**: 基于历史数据分布，黑天鹅事件可能未被包含
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

        # MC路径可视化
        st.markdown("#### 🛤️ 模拟路径")
        st.caption("展示蒙特卡洛模拟的典型路径走势及95%置信区间。")
        st.plotly_chart(plot_mc_paths(mc, n_paths=50, theme=theme), use_container_width=True)

        st.info(f"""
        **模拟结果解读**:
        - 模拟了{n_sim:,}条可能路径，预测{horizon}天后的利差变化
        - 99%概率下，损失不超过{mc['var_99']:.4f}
        - 如果发生极端情况，平均损失为{mc['es_99']:.4f}
        """)

    # 敏感性分析
    st.divider()
    st.markdown("### 🔬 敏感性分析")
    st.caption("评估关键参数变化对VaR/ES风险指标的影响程度。")

    sa_col1, sa_col2 = st.columns(2)
    with sa_col1:
        sa_param = st.selectbox("分析参数", ["volatility", "mean", "df"],
                                format_func=lambda x: {
                                    'volatility': '波动率 σ',
                                    'mean': '均值 μ',
                                    'df': '自由度 df'
                                }.get(x, x))
    with sa_col2:
        if st.button("运行敏感性分析", type="primary"):
            st.session_state['sensitivity'] = run_sensitivity_analysis(returns, param=sa_param)

    if 'sensitivity' in st.session_state:
        sa_df = st.session_state['sensitivity']
        st.plotly_chart(plot_sensitivity_analysis(sa_df, param=sa_param, theme=theme), use_container_width=True)

        # 敏感性摘要
        if len(sa_df) > 1:
            var_range = sa_df['var_99'].max() - sa_df['var_99'].min()
            es_range = sa_df['es_99'].max() - sa_df['es_99'].min()
            sa_sum = st.columns(2)
            with sa_sum[0]:
                safe_metric("VaR变化幅度", var_range, help_text="参数全范围内VaR的最大变化量")
            with sa_sum[1]:
                safe_metric("ES变化幅度", es_range, help_text="参数全范围内ES的最大变化量")

render_app_footer()