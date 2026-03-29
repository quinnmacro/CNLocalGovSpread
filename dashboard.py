"""
中国地方债利差分析仪表板
Streamlit 现代化版本

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
    plot_multi_tenor_spread,
    plot_tenor_spread_correlation,
    plot_tenor_spread_statistics
)
from export import export_to_excel

# ============================================================================
# 工具函数
# ============================================================================

def safe_format(value, unit="", decimals=2):
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

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stMetric > label {
        font-size: 0.9rem;
        color: #666;
    }
    .stMetric > div {
        font-size: 1.5rem;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<p class="main-header">📊 中国地方政府债券利差分析</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Econometric Framework for China Local Government Bond Spread</p>', unsafe_allow_html=True)

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
            st.success(f"✓ 数据加载完成: {len(clean_data)} 个交易日")
        except Exception as e:
            st.error(f"数据加载失败: {str(e)}")
            st.stop()

    # 数据概览卡片
    current_spread = clean_data['spread'].iloc[-1]
    spread_mean = clean_data['spread'].mean()
    spread_std = clean_data['spread'].std()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        safe_metric("当前利差", current_spread, "", "最新交易日利差值")
    with col2:
        safe_metric("历史均值", spread_mean, "", "全样本平均")
    with col3:
        safe_metric("历史标准差", spread_std, "", "波动程度")
    with col4:
        spread_range = clean_data['spread'].max() - clean_data['spread'].min()
        safe_metric("利差区间", spread_range, "", "最大值-最小值")

    st.divider()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 信号分析",
        "📉 波动率分析",
        "⚠️ 风险分析",
        "📊 多期限对比"
    ])

    # =========================================================================
    # Tab 1: 信号分析
    # =========================================================================
    with tab1:
        st.subheader("卡尔曼滤波信号提取")

        with st.spinner("拟合卡尔曼滤波..."):
            try:
                kalman = KalmanSignalExtractor(clean_data['spread'])
                smoothed = kalman.fit()
                deviation = kalman.get_signal_deviation()
            except Exception as e:
                st.error(f"卡尔曼滤波拟合失败: {str(e)}")
                st.stop()

        # 信号图
        fig1 = plot_signal_trend(clean_data, smoothed, deviation)
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
                st.warning(f"🔴 **做空信号**: 利差高估 {dev_val:.2f}σ，预期收窄")
            elif dev_val < -1.5:
                st.success(f"🟢 **做多信号**: 利差低估 {dev_val:.2f}σ，预期扩大")
            else:
                st.info(f"🟡 **中性**: 利差在合理区间，偏离度 {dev_val:.2f}σ")
        else:
            st.info("⚪ 数据平稳，无明显交易信号")

    # =========================================================================
    # Tab 2: 波动率分析
    # =========================================================================
    with tab2:
        st.subheader("GARCH模型锦标赛")

        with st.spinner("运行GARCH锦标赛..."):
            try:
                vol_modeler = VolatilityModeler(returns)
                winner = vol_modeler.run_tournament()
                winner_vol = vol_modeler.get_conditional_volatility(winner)
            except Exception as e:
                st.error(f"波动率建模失败: {str(e)}")
                st.stop()

        # 波动率图
        fig2 = plot_volatility_structure(winner_vol, winner)
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

        # 状态切换（仅在数据足够时）
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
        st.subheader("极值理论(EVT)风险分析")

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
            except Exception as e:
                st.error(f"EVT分析失败: {str(e)}")
                var = returns.quantile(var_confidence)
                es = returns.quantile(0.999)
                st.warning(f"已回退到经验分位数方法")

        # 尾部风险图
        try:
            fig3, evt_var_out, empirical_var = plot_tail_risk(returns, var, var_confidence)
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

        # Hill估计量（可选）
        try:
            with st.spinner("计算Hill估计量..."):
                hill_index = evt.estimate_hill()

            if evt.hill_estimator and evt.gpd_params:
                st.markdown("#### 参数对比")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**GPD形状参数 ξ**: {evt.gpd_params['shape']:.4f}")
                with col2:
                    st.write(f"**Hill形状参数 ξ**: {evt.hill_estimator['shape']:.4f}")
        except:
            pass

    # =========================================================================
    # Tab 4: 多期限对比
    # =========================================================================
    with tab4:
        st.subheader("多期限利差对比分析")

        if data_source != 'CSV':
            st.warning("⚠️ 多期限分析仅支持 CSV 数据源")
        else:
            csv_path = config['CSV_PATH']
            if not os.path.exists(csv_path):
                st.error(f"数据文件不存在: {csv_path}")
            else:
                multi_df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')

                # 多期限趋势图
                st.markdown("#### 各期限利差趋势")
                fig_multi = plot_multi_tenor_spread(multi_df)
                st.plotly_chart(fig_multi, use_container_width=True)

                # 统计对比
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 统计分布对比")
                    fig_box = plot_tenor_spread_statistics(multi_df)
                    st.plotly_chart(fig_box, use_container_width=True)

                with col2:
                    st.markdown("#### 相关性矩阵")
                    fig_corr = plot_tenor_spread_correlation(multi_df)
                    st.plotly_chart(fig_corr, use_container_width=True)

                # 统计摘要表
                st.markdown("#### 统计摘要")
                col_names = {
                    'spread_all': '综合利差',
                    'spread_5y': '5年期',
                    'spread_10y': '10年期',
                    'spread_30y': '30年期'
                }
                summary_data = []
                for col, name in col_names.items():
                    if col in multi_df.columns:
                        series = multi_df[col].dropna()
                        summary_data.append({
                            '期限': name,
                            '均值': f"{series.mean():.4f}",
                            '中位数': f"{series.median():.4f}",
                            '标准差': f"{series.std():.4f}",
                            '最小值': f"{series.min():.4f}",
                            '最大值': f"{series.max():.4f}",
                            '当前值': f"{series.iloc[-1]:.4f}"
                        })
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

                # 期限结构分析
                if all(c in multi_df.columns for c in ['spread_5y', 'spread_10y', 'spread_30y']):
                    st.markdown("#### 期限结构分析")
                    spread_10y_5y = multi_df['spread_10y'] - multi_df['spread_5y']
                    spread_30y_10y = multi_df['spread_30y'] - multi_df['spread_10y']

                    col1, col2 = st.columns(2)
                    with col1:
                        safe_metric("10Y-5Y期限利差", spread_10y_5y.iloc[-1], "",
                                   f"均值: {spread_10y_5y.mean():.4f}")
                    with col2:
                        safe_metric("30Y-10Y期限利差", spread_30y_10y.iloc[-1], "",
                                   f"均值: {spread_30y_10y.mean():.4f}")

                    if spread_10y_5y.iloc[-1] > spread_10y_5y.mean():
                        st.info("📊 期限结构趋陡（长端利差扩大）")
                    else:
                        st.info("📊 期限结构趋平（长端利差收窄）")

# ============================================================================
# 页脚
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1rem;'>
    <small>
        <strong>CNLocalGovSpread</strong> v1.3.0 |
        Author: <a href='https://github.com/quinnmacro' target='_blank'>Quinn Liu</a> |
        <a href='https://github.com/quinnmacro/CNLocalGovSpread' target='_blank'>GitHub</a> |
        <a href='https://www.linkedin.com/in/liulu-math' target='_blank'>LinkedIn</a>
    </small>
</div>
""", unsafe_allow_html=True)
