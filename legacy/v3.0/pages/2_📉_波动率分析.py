"""
波动率分析页面 - GARCH模型锦标赛 + ML模型对比 + FIGARCH长记忆 + 参数校准
v3.0增强: 集成ML波动率模型、FIGARCH长记忆检测、参数自动校准
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

render_page_header("📉 波动率分析", "GARCH Tournament | ML Comparison | FIGARCH | Calibration")

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
    **📖 波动率分析**: GARCH模型锦标赛 + ML模型对比 + 长记忆检测 + 参数校准

    • **GARCH锦标赛**: 传统计量经济学模型（GARCH/EGARCH/GJR-GARCH/EWMA），AIC/BIC选择最优
    • **ML模型对比**: Random Forest/XGBoost/LSTM，数据驱动波动率预测
    • **FIGARCH**: 长记忆波动率检测，捕捉波动率持续性超GARCH范围的行为
    • **参数校准**: 从数据自动估计lambda/df/phi等关键参数，替代硬编码默认值
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
    st.markdown("#### 📊 GARCH模型对比与选择")
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
    # v3.0新增: FIGARCH长记忆检测
    # =========================================================================
    st.divider()
    st.markdown("### 🔍 FIGARCH长记忆检测")

    lm_result = results.get('long_memory')
    figarch_result = results.get('figarch')

    if lm_result:
        d_est = lm_result.get('d_estimate', 0)
        d_se = lm_result.get('d_std_error', 0)
        d_pval = lm_result.get('d_p_value', 1)
        memory_type = lm_result.get('memory_type', '未知')

        lm_cols = st.columns(3)
        with lm_cols[0]:
            safe_metric("长记忆参数 d", d_est, help_text="GPH估计的分数差分参数")
        with lm_cols[1]:
            safe_metric("显著性 p值", d_pval, help_text="d参数的统计显著性")
        with lm_cols[2]:
            st.metric("记忆类型", memory_type)

        memory_desc = {
            '短记忆': "**短记忆 (d≈0)**: 波动率冲击快速衰减，GARCH框架充分。无需FIGARCH。",
            '长记忆': "**长记忆 (0<d<0.5)**: 波动率冲击持续衰减，FIGARCH可捕捉超GARCH持续性。建议关注FIGARCH模型。",
            '中长记忆': "**中长记忆 (0.5≤d<1)**: 强持续性，FIGARCH模型显著优于GARCH。波动率预测需考虑长记忆效应。",
            '无限记忆': "**无限记忆 (d≈1)**: 极端持续性，波动率几乎不衰减。需特别关注风险累积。"
        }
        desc = memory_desc.get(memory_type, memory_type)
        if d_pval < 0.05:
            st.warning(f"📊 长记忆检测: d={d_est:.4f} (p={d_pval:.4f})\n\n{desc}")
        else:
            st.success(f"📊 长记忆检测: d={d_est:.4f} (p={d_pval:.4f})\n\n{desc}")

        if figarch_result:
            figarch_aic = figarch_result.get('AIC', float('inf'))
            figarch_bic = figarch_result.get('BIC', float('inf'))
            st.caption(f"FIGARCH(1,d,1) AIC={figarch_aic:.1f}, BIC={figarch_bic:.1f} | d={figarch_result.get('d', d_est):.4f}")
    else:
        st.info("长记忆检测暂不可用，请检查数据是否充足。")

    # =========================================================================
    # v3.0新增: ML波动率模型对比
    # =========================================================================
    st.divider()
    st.markdown("### 🤖 ML波动率模型对比")

    ml_comparison = results.get('ml_comparison')
    ml_winner = results.get('ml_winner')
    garch_comparison = results.get('garch_comparison')

    if ml_comparison:
        ml_cols = st.columns(3)
        with ml_cols[0]:
            st.metric("ML获胜模型", ml_winner or "N/A")
        with ml_cols[1]:
            if ml_winner and ml_winner in ml_comparison:
                ml_rmse = ml_comparison[ml_winner].get('RMSE', None)
            else:
                ml_rmse = None
            safe_metric("ML RMSE", ml_rmse, help_text="ML最优模型的RMSE")
        with ml_cols[2]:
            safe_metric("综合获胜", results.get('garch_overall_winner', 'N/A'),
                       help_text="GARCH+ML全模型综合AIC获胜者")

        # 全模型对比表 (GARCH vs ML)
        if garch_comparison is not None and isinstance(garch_comparison, pd.DataFrame):
            display_df = garch_comparison.copy()
            display_df = display_df.reset_index()
            display_df.columns = ['模型', 'AIC', 'BIC', 'RMSE', 'MAE', '类型']
            # 格式化数值
            for col in ['AIC', 'BIC', 'RMSE', 'MAE']:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) and not pd.isinf(x) else 'N/A'
                )
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            garch_count = len(vol_modeler.ic_scores)
            ml_count = len(ml_comparison)
            st.caption(f"模型总数: {garch_count + ml_count} (GARCH: {garch_count}, ML: {ml_count})")
        else:
            # 仅显示ML结果
            ml_data = []
            for m, metrics in ml_comparison.items():
                ml_data.append({
                    '模型': m,
                    'RMSE': f"{metrics.get('RMSE', 'N/A')}",
                    'MAE': f"{metrics.get('MAE', 'N/A')}",
                    'AIC': f"{metrics.get('AIC', 'N/A')}"
                })
            st.dataframe(pd.DataFrame(ml_data), use_container_width=True, hide_index=True)
    else:
        st.info("ML模型对比暂不可用。Random Forest需要sklearn库支持。")

    # =========================================================================
    # v3.0新增: 参数自动校准
    # =========================================================================
    st.divider()
    st.markdown("### ⚙️ 参数自动校准")

    calibrated = results.get('calibrated')
    calibrator = results.get('calibrator')

    if calibrated:
        cal_cols = st.columns(3)
        with cal_cols[0]:
            safe_metric("EWMA λ", calibrated.get('ewma_lambda'),
                       help_text="QLIKE优化估计的EWMA衰减因子")
        with cal_cols[1]:
            safe_metric("t分布 df", calibrated.get('t_df'),
                       help_text="MLE估计的Student-t自由度参数")
        with cal_cols[2]:
            safe_metric("AR(1) φ", calibrated.get('ar_phi'),
                       help_text="OLS估计的利差变化AR系数")

        # 校准参数对比默认值
        param_data = []
        default_vs_calibrated = {
            'ewma_lambda': ('EWMA λ', 0.94),
            't_df': ('t分布 df', 5.0),
            'ar_phi': ('AR(1) φ', 0.98),
            'evt_threshold_percentile': ('EVT阈值%', 0.95),
            'kalman_window': ('卡尔曼窗口', 60),
            'signal_threshold': ('信号阈值σ', 1.5),
        }
        for key, (name, default) in default_vs_calibrated.items():
            cal_val = calibrated.get(key)
            if cal_val is not None:
                diff_pct = ((cal_val - default) / default) * 100 if default != 0 else 0
                param_data.append({
                    '参数': name,
                    '默认值': f"{default}",
                    '校准值': f"{cal_val:.4f}",
                    '偏差': f"{diff_pct:+.1f}%"
                })
        if param_data:
            st.dataframe(pd.DataFrame(param_data), use_container_width=True, hide_index=True)
            st.caption("参数校准使用数据驱动估计替代硬编码默认值，提高模型适配性。")

        # GARCH持续性诊断
        if calibrator:
            try:
                persistence_diag = calibrator.diagnose_garch_persistence()
                if persistence_diag:
                    st.markdown("**GARCH持续性诊断**")
                    diag_cols = st.columns(2)
                    with diag_cols[0]:
                        safe_metric("α+β (GARCH)", persistence_diag.get('alpha_beta_sum', None),
                                   help_text="GARCH持续性参数之和，接近1表示高度持续")
                    with diag_cols[1]:
                        safe_metric("持续性判定", persistence_diag.get('persistence_level', 'N/A'))
            except:
                pass
    else:
        st.info("参数校准暂不可用，请检查数据是否充足。")

render_app_footer()