"""
可视化模块 - 生成交互式图表

使用 Plotly 生成三张专业图表：
1. 信号与趋势图 - 卡尔曼滤波 vs 原始利差
2. 波动率结构图 - 条件波动率 + 危机模式识别
3. 尾部风险锥 - 收益率分布 + Student-t 拟合 + VaR
"""

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go


def plot_signal_trend(clean_data, smoothed_spread, signal_deviation):
    """
    图表 1: 信号与趋势 (卡尔曼滤波 vs 原始利差)
    """
    # 识别交易信号点（偏离 > 1.5σ）
    buy_signals = signal_deviation[signal_deviation < -1.5]
    sell_signals = signal_deviation[signal_deviation > 1.5]

    fig = go.Figure()

    # 原始利差（灰色半透明）
    fig.add_trace(go.Scatter(
        x=clean_data.index,
        y=clean_data['spread'],
        mode='lines',
        name='原始利差',
        line=dict(color='lightgray', width=1),
        opacity=0.5
    ))

    # Kalman 平滑利差（蓝色粗线）
    fig.add_trace(go.Scatter(
        x=smoothed_spread.index,
        y=smoothed_spread.values,
        mode='lines',
        name='卡尔曼趋势',
        line=dict(color='#1f77b4', width=2.5)
    ))

    # 买入信号（绿色向上三角）
    if len(buy_signals) > 0:
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=clean_data.loc[buy_signals.index, 'spread'],
            mode='markers',
            name='买入信号 (低估)',
            marker=dict(symbol='triangle-up', size=10, color='green')
        ))

    # 卖出信号（红色向下三角）
    if len(sell_signals) > 0:
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=clean_data.loc[sell_signals.index, 'spread'],
            mode='markers',
            name='卖出信号 (高估)',
            marker=dict(symbol='triangle-down', size=10, color='red')
        ))

    fig.update_layout(
        title='图表 1: 信号提取 - 卡尔曼滤波 vs 原始利差',
        xaxis_title='日期',
        yaxis_title='利差 (bps)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )

    return fig


def plot_volatility_structure(winner_volatility, winner_model):
    """
    图表 2: 波动率结构（锦标赛获胜模型）
    """
    # 计算波动率的 90% 分位数（高波动阈值）
    vol_threshold = winner_volatility.quantile(0.90)
    high_vol_periods = winner_volatility[winner_volatility > vol_threshold]

    fig = go.Figure()

    # 条件波动率曲线
    fig.add_trace(go.Scatter(
        x=winner_volatility.index,
        y=winner_volatility.values,
        mode='lines',
        name=f'{winner_model} 条件波动率',
        line=dict(color='purple', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(128, 0, 128, 0.1)'
    ))

    # 高波动阈值线（虚线）
    fig.add_hline(
        y=vol_threshold,
        line_dash='dash',
        line_color='red',
        annotation_text=f'危机模式阈值 (90%分位: {vol_threshold:.2f} bps)',
        annotation_position='right'
    )

    # 标记高波动期（红色点）
    if len(high_vol_periods) > 0:
        fig.add_trace(go.Scatter(
            x=high_vol_periods.index,
            y=high_vol_periods.values,
            mode='markers',
            name='高波动期 (危机模式)',
            marker=dict(color='red', size=6, symbol='diamond')
        ))

    fig.update_layout(
        title=f'图表 2: 波动率结构 - {winner_model} 模型',
        xaxis_title='日期',
        yaxis_title='条件波动率 (bps)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )

    return fig


def plot_tail_risk(returns, evt_var, confidence=0.99):
    """
    图表 3: 尾部风险锥（收益率分布 + Student-t 拟合 + VaR）
    """
    fig = go.Figure()

    # 直方图（实际收益率分布）
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='实际分布',
        marker_color='lightblue',
        opacity=0.7,
        histnorm='probability density'
    ))

    # 拟合 Student-t 分布曲线
    df_fit, loc_fit, scale_fit = stats.t.fit(returns)
    x_range = np.linspace(returns.min(), returns.max(), 200)
    t_pdf = stats.t.pdf(x_range, df_fit, loc_fit, scale_fit)

    fig.add_trace(go.Scatter(
        x=x_range,
        y=t_pdf,
        mode='lines',
        name=f'Student-t 拟合 (df={df_fit:.1f})',
        line=dict(color='darkblue', width=2.5)
    ))

    # 正态分布对比线（展示肥尾效应）
    normal_pdf = stats.norm.pdf(x_range, returns.mean(), returns.std())
    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal_pdf,
        mode='lines',
        name='正态分布对比',
        line=dict(color='orange', width=2, dash='dot'),
        opacity=0.6
    ))

    # VaR 标记线（红色虚线）
    fig.add_vline(
        x=evt_var,
        line_dash='dash',
        line_color='red',
        line_width=3,
        annotation_text=f'99% EVT-VaR: {evt_var:.2f} bps',
        annotation_position='top'
    )

    # 也标记经验分位数（对比）
    empirical_var = returns.quantile(confidence)
    fig.add_vline(
        x=empirical_var,
        line_dash='dot',
        line_color='gray',
        annotation_text=f'经验99%分位: {empirical_var:.2f} bps',
        annotation_position='bottom'
    )

    fig.update_layout(
        title='图表 3: 尾部风险锥 - 收益率分布与 VaR',
        xaxis_title='利差变化 (bps)',
        yaxis_title='概率密度',
        hovermode='x',
        template='plotly_white',
        height=500,
        showlegend=True
    )

    return fig, evt_var, empirical_var


def print_var_comparison(evt_var, empirical_var):
    """打印VaR对比分析"""
    print(f"\n对比分析:")
    print(f"  EVT-VaR (99%):    {evt_var:.2f} bps")
    print(f"  经验分位数 (99%): {empirical_var:.2f} bps")
    print(f"  差异:             {evt_var - empirical_var:.2f} bps")
    if evt_var > empirical_var:
        print(f"  ⚠️  EVT 估计的风险更高（更保守），这是肥尾效应的体现")
    else:
        print(f"  ℹ️  EVT 与经验分位数接近，尾部风险可控")
