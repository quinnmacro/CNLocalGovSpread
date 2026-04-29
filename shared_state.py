"""
共享状态管理模块 - 多页Dashboard共用

功能:
1. 页面配置初始化 (page config, theme)
2. 侧边栏配置渲染
3. 分析数据加载与缓存 (session state)
4. 公共工具函数 (safe_format, safe_metric)
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from data_engine import DataEngine
from volatility import VolatilityModeler, RegimeDetector
from kalman import KalmanSignalExtractor
from evt import EVTRiskAnalyzer
from visualization import (
    plot_signal_trend,
    plot_volatility_structure,
    plot_tail_risk
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
    calculate_rolling_stats,
    detect_historical_events,
    plot_rolling_stats,
    plot_percentile_chart
)
from alerts import check_risk_alerts, get_risk_score
from report_gen import generate_report, get_report_history, generate_quick_report
from content import (
    render_quick_reference,
    render_report_guide,
    render_metric_interpretation,
    render_trading_advice,
    get_spread_position_comment,
    get_volatility_comment,
    get_var_comment
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
# 页面初始化
# ============================================================================

def init_page(page_title="中国地方债利差分析", page_icon="📊"):
    """初始化页面配置和主题"""
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 主题切换 (默认深色)
    theme = get_theme_toggle()
    apply_theme(theme)
    return theme


# ============================================================================
# 侧边栏配置
# ============================================================================

def render_sidebar():
    """渲染侧边栏配置，返回配置字典"""
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

    return config, run_analysis


# ============================================================================
# 数据加载与分析逻辑
# ============================================================================

def ensure_analysis(config, run_analysis):
    """确保分析已完成，如需要则执行数据加载和模型拟合"""
    need_load = 'analysis_done' not in st.session_state or not st.session_state.analysis_done

    if need_load:
        load_placeholder = st.status("⚡ 正在初始化分析引擎...", expanded=True)

    if run_analysis or need_load:
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
            evt = EVTRiskAnalyzer(returns, threshold_percentile=config['EVT_THRESHOLD_PERCENTILE'],
                                  confidence=config['VaR_CONFIDENCE'])
            evt.fit_gpd()
            var = evt.calculate_var()
            es = evt.calculate_es()
            st.session_state.evt = evt
            st.session_state.var = var
            st.session_state.es = es
        except Exception as e:
            st.session_state.evt = None
            st.session_state.var = returns.quantile(config['VaR_CONFIDENCE'])
            st.session_state.es = returns.quantile(0.999)

        st.session_state.analysis_done = True

        if need_load:
            load_placeholder.update(label="✅ 分析完成!", state="complete")


def get_results():
    """获取分析结果，返回所有关键数据"""
    if not st.session_state.get('analysis_done', False):
        return None

    return {
        'clean_data': st.session_state.clean_data,
        'returns': st.session_state.returns,
        'kalman': st.session_state.kalman,
        'smoothed': st.session_state.smoothed,
        'deviation': st.session_state.deviation,
        'vol_modeler': st.session_state.vol_modeler,
        'winner': st.session_state.winner,
        'winner_vol': st.session_state.winner_vol,
        'evt': st.session_state.evt,
        'var': st.session_state.var,
        'es': st.session_state.es,
    }


def render_app_footer(version='3.0.0'):
    """渲染页脚 - 使用默认参数包装 styles.render_footer"""
    render_footer(
        version=version,
        author='Quinn Liu',
        github='https://github.com/quinnmacro/CNLocalGovSpread',
        linkedin='https://www.linkedin.com/in/liulu-math'
    )