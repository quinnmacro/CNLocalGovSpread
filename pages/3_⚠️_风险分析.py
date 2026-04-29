"""
风险分析页面 - 极值理论VaR/ES
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared_state import (
    init_page, render_sidebar, ensure_analysis,
    get_results, safe_metric, render_app_footer,
    plot_tail_risk, render_metric_interpretation,
    render_page_header
)

# ============================================================================
# 页面初始化
# ============================================================================

theme = init_page(page_title="风险分析", page_icon="⚠️")
config, run_analysis = render_sidebar()
ensure_analysis(config, run_analysis)

render_page_header("⚠️ 风险分析", "Extreme Value Theory - VaR & ES")

# ============================================================================
# 风险分析内容
# ============================================================================

results = get_results()
if results:
    returns = results['returns']
    evt = results['evt']
    var = results['var']
    es = results['es']
    var_confidence = config['VaR_CONFIDENCE']

    # 核心说明
    st.info("""
    **📖 风险分析**: 极值理论(EVT)专注于尾部风险量化

    • **VaR (利差扩大风险)**: 给定置信度下，利差扩大的最大预期值（右尾上侧分位数）
    • **ES (预期损失)**: 超越VaR时的平均损失，比VaR更保守，Basel III已采用
    • **尾部指数 ξ**: ξ > 0 表示厚尾分布，极端损失概率高于正态预测
    • **样本比例因子**: VaR计算已包含 n/N_u 因子，正确估计超越阈值的概率
    • **为什么不用正态分布?** 正态分布严重低估极端损失概率，EVT更可靠
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

render_app_footer()