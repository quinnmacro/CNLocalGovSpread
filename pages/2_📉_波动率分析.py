"""
波动率分析页面 - GARCH模型锦标赛与状态切换
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared_state import (
    init_page, render_sidebar, ensure_analysis,
    get_results, safe_metric, render_app_footer,
    plot_volatility_structure, render_metric_interpretation,
    VolatilityModeler, RegimeDetector, render_page_header
)
import pandas as pd

# ============================================================================
# 页面初始化
# ============================================================================

theme = init_page(page_title="波动率分析", page_icon="📉")
config, run_analysis = render_sidebar()
ensure_analysis(config, run_analysis)

render_page_header("📉 波动率分析", "GARCH Model Tournament & Regime Detection")

# ============================================================================
# 波动率分析内容
# ============================================================================

results = get_results()
if results:
    winner = results['winner']
    winner_vol = results['winner_vol']
    vol_modeler = results['vol_modeler']
    current_vol = winner_vol.iloc[-1]

    # 核心说明
    st.info("""
    **📖 波动率分析**: GARCH模型锦标赛，让数据选择最优模型

    • **波动率聚集**: 平静期后往往平静，动荡期后往往动荡（波动率具有持续性）
    • **Student-t分布**: 所有模型统一使用t分布，捕捉厚尾特征（极端事件更频繁）
    • **模型选择**: AIC越小越好（EWMA使用t分布似然，与GARCH可比）
    • **状态切换**: HMM识别低/中/高波动率状态，辅助仓位决策
    • **注**: EGARCH在arch库中为对称模型，非对称分析请参考GJR-GARCH
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

render_app_footer()