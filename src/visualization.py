"""
可视化模块 - 生成交互式图表

使用 Plotly 生成专业图表：
1. 信号与趋势图 - 卡尔曼滤波 vs 原始利差 (带区间选择器)
2. 波动率结构图 - 条件波动率 + 危机模式识别
3. 尾部风险锥 - 收益率分布 + Student-t 拟合 + VaR
4. 多期限对比图
5. 风险预警图
"""

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================================
# 区间选择器和主题配置
# ============================================================================

def add_range_selector(fig, dark_mode=False):
    """添加日期区间选择器"""
    button_color = '#FAFAFA' if dark_mode else '#1E3A5F'
    bg_color = 'rgba(30,30,30,0.8)' if dark_mode else 'rgba(255,255,255,0.8)'

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1月", step="month", stepmode="backward"),
                    dict(count=3, label="3月", step="month", stepmode="backward"),
                    dict(count=6, label="6月", step="month", stepmode="backward"),
                    dict(count=1, label="1年", step="year", stepmode="backward"),
                    dict(count=3, label="3年", step="year", stepmode="backward"),
                    dict(step="all", label="全部")
                ],
                font=dict(color=button_color, size=11),
                bgcolor=bg_color,
                activecolor='#667eea'
            ),
            rangeslider=dict(visible=True, thickness=0.08),
            type="date"
        )
    )
    return fig


def get_theme_config(theme='light'):
    """获取主题配置 - 现代化透明背景"""
    if theme == 'dark':
        return {
            'template': 'plotly_dark',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font_color': '#F8FAFC',
            'grid_color': 'rgba(71, 85, 105, 0.3)',
            'line_color': 'rgba(71, 85, 105, 0.5)',
            'colors': ['#3B82F6', '#8B5CF6', '#06B6D4', '#10B981', '#F59E0B', '#EF4444']
        }
    return {
        'template': 'none',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font_color': '#0F172A',
        'grid_color': 'rgba(226, 232, 240, 0.8)',
        'line_color': 'rgba(203, 213, 225, 0.8)',
        'colors': ['#2563EB', '#7C3AED', '#0891B2', '#059669', '#D97706', '#DC2626']
    }


# ============================================================================
# 信号趋势图 (增强版)
# ============================================================================

def plot_signal_trend(clean_data, smoothed_spread, signal_deviation, theme='light'):
    """
    图表 1: 信号与趋势 (卡尔曼滤波 vs 原始利差) - 增强版

    参数:
        clean_data: 清洗后的数据
        smoothed_spread: 平滑后的利差
        signal_deviation: 信号偏离度
        theme: 'light' 或 'dark'
    """
    config = get_theme_config(theme)
    dark_mode = theme == 'dark'

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
        line=dict(color='#94a3b8', width=1),
        opacity=0.6
    ))

    # Kalman 平滑利差（蓝色粗线）
    fig.add_trace(go.Scatter(
        x=smoothed_spread.index,
        y=smoothed_spread.values,
        mode='lines',
        name='卡尔曼趋势',
        line=dict(color='#3b82f6', width=2.5)
    ))

    # 置信区间 - P0修复: 标注为"残差参考区间"而非统计置信区间
    # 注: 此区间基于残差样本标准差，非Kalman滤波器的状态协方差
    std = (clean_data['spread'] - smoothed_spread).std()
    fig.add_trace(go.Scatter(
        x=smoothed_spread.index,
        y=smoothed_spread.values + 1.5 * std,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=smoothed_spread.index,
        y=smoothed_spread.values - 1.5 * std,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.15)',
        name='参考区间 (±1.5σ, 残差)',
        hoverinfo='skip'
    ))

    # 买入信号（绿色向上三角）- P0修复: 添加索引对齐检查
    if len(buy_signals) > 0:
        # 确保索引对齐
        common_index = buy_signals.index.intersection(clean_data.index)
        if len(common_index) > 0:
            fig.add_trace(go.Scatter(
                x=common_index,
                y=clean_data.loc[common_index, 'spread'],
                mode='markers',
                name='买入信号 (低估)',
                marker=dict(symbol='triangle-up', size=12, color='#22c55e', line=dict(width=2, color='white'))
            ))

    # 卖出信号（红色向下三角）- P0修复: 添加索引对齐检查
    if len(sell_signals) > 0:
        # 确保索引对齐
        common_index = sell_signals.index.intersection(clean_data.index)
        if len(common_index) > 0:
            fig.add_trace(go.Scatter(
                x=common_index,
                y=clean_data.loc[common_index, 'spread'],
                mode='markers',
                name='卖出信号 (高估)',
                marker=dict(symbol='triangle-down', size=12, color='#ef4444', line=dict(width=2, color='white'))
            ))

    fig.update_layout(
        title='信号提取 - 卡尔曼滤波分析',
        xaxis_title='日期',
        yaxis_title='利差',
        hovermode='x unified',
        template=config['template'],
        height=550,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        paper_bgcolor=config['paper_bgcolor'],
        plot_bgcolor=config['plot_bgcolor'],
        font=dict(color=config['font_color'])
    )

    fig.update_xaxes(gridcolor=config['grid_color'], linecolor=config['line_color'])
    fig.update_yaxes(gridcolor=config['grid_color'], linecolor=config['line_color'])

    # 添加区间选择器
    fig = add_range_selector(fig, dark_mode)

    return fig


def plot_volatility_structure(winner_volatility, winner_model, theme='light'):
    """
    图表 2: 波动率结构（锦标赛获胜模型）- 增强版

    参数:
        winner_volatility: 条件波动率序列
        winner_model: 获胜模型名称
        theme: 'light' 或 'dark'
    """
    config = get_theme_config(theme)

    # P0修复: 标注前视偏差
    # 计算波动率的 90% 分位数（高波动阈值）
    # 注: 此阈值基于全样本后验计算，实际交易中无法预知未来数据，仅供展示参考
    vol_threshold = winner_volatility.quantile(0.90)
    high_vol_periods = winner_volatility[winner_volatility > vol_threshold]

    fig = go.Figure()

    # 条件波动率曲线（渐变填充）
    fig.add_trace(go.Scatter(
        x=winner_volatility.index,
        y=winner_volatility.values,
        mode='lines',
        name=f'{winner_model} 条件波动率',
        line=dict(color='#8b5cf6', width=2),
        fill='tozeroy',
        fillcolor='rgba(139, 92, 246, 0.15)'
    ))

    # 高波动阈值线（虚线）
    fig.add_hline(
        y=vol_threshold,
        line_dash='dash',
        line_color='#ef4444',
        line_width=2,
        annotation_text=f'危机模式阈值 (90%分位: {vol_threshold:.4f})',
        annotation_position='right',
        annotation_font=dict(color='#ef4444', size=11)
    )

    # 标记高波动期（红色菱形点）
    if len(high_vol_periods) > 0:
        fig.add_trace(go.Scatter(
            x=high_vol_periods.index,
            y=high_vol_periods.values,
            mode='markers',
            name='高波动期 (危机模式)',
            marker=dict(
                color='#ef4444',
                size=8,
                symbol='diamond',
                line=dict(width=1, color='white')
            ),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>波动率: %{y:.4f}<extra></extra>'
        ))

    fig.update_layout(
        title=f'波动率结构分析 - {winner_model} 模型',
        xaxis_title='日期',
        yaxis_title='条件波动率',
        hovermode='x unified',
        template=config['template'],
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        paper_bgcolor=config['paper_bgcolor'],
        plot_bgcolor=config['plot_bgcolor'],
        font=dict(color=config['font_color'])
    )

    fig.update_xaxes(gridcolor=config['grid_color'], linecolor=config['line_color'])
    fig.update_yaxes(gridcolor=config['grid_color'], linecolor=config['line_color'])

    # 添加区间选择器
    fig = add_range_selector(fig, theme == 'dark')

    return fig


def plot_tail_risk(returns, evt_var, confidence=0.99, theme='light'):
    """
    图表 3: 尾部风险锥（收益率分布 + Student-t 拟合 + VaR）- 增强版

    参数:
        returns: 收益率序列
        evt_var: EVT计算的VaR值
        confidence: 置信水平
        theme: 'light' 或 'dark'
    """
    config = get_theme_config(theme)

    fig = go.Figure()

    # 直方图（实际收益率分布）
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='实际分布',
        marker_color='#60a5fa',
        opacity=0.7,
        histnorm='probability density',
        marker_line=dict(width=0)
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
        line=dict(color='#1d4ed8', width=2.5)
    ))

    # 正态分布对比线（展示肥尾效应）
    normal_pdf = stats.norm.pdf(x_range, returns.mean(), returns.std())
    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal_pdf,
        mode='lines',
        name='正态分布对比',
        line=dict(color='#f59e0b', width=2, dash='dot'),
        opacity=0.7
    ))

    # VaR 标记线（红色虚线）- P0修复: 明确标注为"利差扩大风险"
    fig.add_vline(
        x=evt_var,
        line_dash='dash',
        line_color='#ef4444',
        line_width=3,
        annotation_text=f'99% EVT-VaR (利差扩大风险): {evt_var:.4f}',
        annotation_position='top',
        annotation_font=dict(color='#ef4444', size=11)
    )

    # 也标记经验分位数（对比）- P0修复: 明确标注为"上侧分位"
    empirical_var = returns.quantile(confidence)
    fig.add_vline(
        x=empirical_var,
        line_dash='dot',
        line_color='#6b7280',
        line_width=2,
        annotation_text=f'经验99%上侧分位: {empirical_var:.4f}',
        annotation_position='bottom',
        annotation_font=dict(color='#6b7280', size=10)
    )

    fig.update_layout(
        title='尾部风险锥 - 收益率分布与 VaR',
        xaxis_title='利差变化',
        yaxis_title='概率密度',
        hovermode='x',
        template=config['template'],
        height=500,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        paper_bgcolor=config['paper_bgcolor'],
        plot_bgcolor=config['plot_bgcolor'],
        font=dict(color=config['font_color'])
    )

    fig.update_xaxes(gridcolor=config['grid_color'], linecolor=config['line_color'])
    fig.update_yaxes(gridcolor=config['grid_color'], linecolor=config['line_color'])

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


def plot_multi_tenor_spread(df, columns=None, theme='light'):
    """
    图表: 多期限利差对比 - 增强版

    参数:
        df: DataFrame，包含多期限利差数据
        columns: 列名列表，默认为 ['spread_all', 'spread_5y', 'spread_10y', 'spread_30y']
        theme: 'light' 或 'dark'

    返回:
        plotly Figure 对象
    """
    config = get_theme_config(theme)

    if columns is None:
        columns = ['spread_all', 'spread_5y', 'spread_10y', 'spread_30y']

    # 过滤存在的列
    available_cols = [c for c in columns if c in df.columns]
    if not available_cols:
        raise ValueError("没有可用的利差列")

    # 列名映射
    col_names = {
        'spread_all': '综合利差',
        'spread_5y': '5年期',
        'spread_10y': '10年期',
        'spread_30y': '30年期'
    }

    # 颜色映射（更现代的配色）
    colors = {
        'spread_all': '#3b82f6',   # 蓝色
        'spread_5y': '#22c55e',    # 绿色
        'spread_10y': '#f59e0b',   # 橙色
        'spread_30y': '#ef4444'    # 红色
    }

    fig = go.Figure()

    for col in available_cols:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            mode='lines',
            name=col_names.get(col, col),
            line=dict(color=colors.get(col, '#6b7280'), width=1.8),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>' + col_names.get(col, col) + ': %{y:.4f}<extra></extra>'
        ))

    fig.update_layout(
        title='多期限利差对比分析',
        xaxis_title='日期',
        yaxis_title='利差',
        hovermode='x unified',
        template=config['template'],
        height=500,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        paper_bgcolor=config['paper_bgcolor'],
        plot_bgcolor=config['plot_bgcolor'],
        font=dict(color=config['font_color'])
    )

    fig.update_xaxes(gridcolor=config['grid_color'], linecolor=config['line_color'])
    fig.update_yaxes(gridcolor=config['grid_color'], linecolor=config['line_color'])

    # 添加区间选择器
    fig = add_range_selector(fig, theme == 'dark')

    return fig


def plot_tenor_spread_correlation(df, columns=None, theme='light'):
    """
    图表: 期限利差相关性热力图 - 增强版

    参数:
        df: DataFrame，包含多期限利差数据
        columns: 列名列表
        theme: 'light' 或 'dark'

    返回:
        plotly Figure 对象
    """
    config = get_theme_config(theme)

    if columns is None:
        columns = ['spread_all', 'spread_5y', 'spread_10y', 'spread_30y']

    available_cols = [c for c in columns if c in df.columns]
    if len(available_cols) < 2:
        raise ValueError("至少需要2列数据才能计算相关性")

    # 列名映射
    col_names = {
        'spread_all': '综合',
        'spread_5y': '5Y',
        'spread_10y': '10Y',
        'spread_30y': '30Y'
    }

    # 计算相关性矩阵
    corr_matrix = df[available_cols].corr()

    # 转换为显示名称
    display_names = [col_names.get(c, c) for c in available_cols]

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=display_names,
        y=display_names,
        colorscale='RdBu',
        zmid=0,
        text=[[f'{v:.2f}' for v in row] for row in corr_matrix.values],
        texttemplate='%{text}',
        textfont={'size': 14, 'color': 'white'},
        hoverongaps=False,
        colorbar=dict(
            title=dict(text='相关系数', font=dict(color=config['font_color'])),
            tickfont=dict(color=config['font_color'])
        )
    ))

    fig.update_layout(
        title='期限利差相关性矩阵',
        template=config['template'],
        height=400,
        width=500,
        paper_bgcolor=config['paper_bgcolor'],
        plot_bgcolor=config['plot_bgcolor'],
        font=dict(color=config['font_color'])
    )

    return fig


def plot_tenor_spread_statistics(df, columns=None, theme='light'):
    """
    图表: 期限利差统计对比（箱线图）- 增强版

    参数:
        df: DataFrame，包含多期限利差数据
        columns: 列名列表
        theme: 'light' 或 'dark'

    返回:
        plotly Figure 对象
    """
    config = get_theme_config(theme)

    if columns is None:
        columns = ['spread_all', 'spread_5y', 'spread_10y', 'spread_30y']

    available_cols = [c for c in columns if c in df.columns]

    # 列名映射
    col_names = {
        'spread_all': '综合',
        'spread_5y': '5Y',
        'spread_10y': '10Y',
        'spread_30y': '30Y'
    }

    # 颜色映射
    colors = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444']

    fig = go.Figure()

    for i, col in enumerate(available_cols):
        fig.add_trace(go.Box(
            y=df[col].dropna(),
            name=col_names.get(col, col),
            boxmean='sd',
            marker_color=colors[i % len(colors)],
            boxpoints='outliers',
            jitter=0.3,
            whiskerwidth=0.5,
            fillcolor=colors[i % len(colors)],
            opacity=0.7,
            line=dict(width=2)
        ))

    fig.update_layout(
        title='期限利差统计分布',
        yaxis_title='利差',
        template=config['template'],
        height=400,
        showlegend=False,
        paper_bgcolor=config['paper_bgcolor'],
        plot_bgcolor=config['plot_bgcolor'],
        font=dict(color=config['font_color'])
    )

    fig.update_yaxes(gridcolor=config['grid_color'], linecolor=config['line_color'])

    return fig


def plot_credit_spread_comparison(local_gov_df, credit_df=None, credit_columns=None, theme='light'):
    """
    图表: 信用利差对比分析 - 增强版

    参数:
        local_gov_df: DataFrame，地方债利差数据
        credit_df: DataFrame，信用利差数据（可选）
        credit_columns: 信用利差列名列表
        theme: 'light' 或 'dark'

    返回:
        plotly Figure 对象
    """
    config = get_theme_config(theme)

    fig = go.Figure()

    # 添加地方债综合利差作为基准
    if 'spread_all' in local_gov_df.columns:
        fig.add_trace(go.Scatter(
            x=local_gov_df.index,
            y=local_gov_df['spread_all'],
            mode='lines',
            name='地方债 (基准)',
            line=dict(color='#3b82f6', width=2.5),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>地方债: %{y:.4f}<extra></extra>'
        ))

    # 如果有信用利差数据，添加对比
    if credit_df is not None and credit_columns:
        colors = ['#f59e0b', '#22c55e', '#ef4444', '#8b5cf6']
        for i, col in enumerate(credit_columns):
            if col in credit_df.columns:
                fig.add_trace(go.Scatter(
                    x=credit_df.index,
                    y=credit_df[col],
                    mode='lines',
                    name=col.replace('credit_', '').replace('_', ' ').title(),
                    line=dict(color=colors[i % len(colors)], width=1.8, dash='dot')
                ))

    fig.update_layout(
        title='信用利差对比分析',
        xaxis_title='日期',
        yaxis_title='利差',
        hovermode='x unified',
        template=config['template'],
        height=500,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        paper_bgcolor=config['paper_bgcolor'],
        plot_bgcolor=config['plot_bgcolor'],
        font=dict(color=config['font_color'])
    )

    fig.update_xaxes(gridcolor=config['grid_color'], linecolor=config['line_color'])
    fig.update_yaxes(gridcolor=config['grid_color'], linecolor=config['line_color'])

    # 添加区间选择器
    fig = add_range_selector(fig, theme == 'dark')

    return fig


def plot_spread_premium_analysis(local_gov_df, credit_df=None, credit_column='credit_corp_aaa', theme='light'):
    """
    图表: 信用利差溢价分析 - 增强版

    分析信用利差相对于地方债利差的溢价

    参数:
        local_gov_df: DataFrame，地方债利差数据
        credit_df: DataFrame，信用利差数据
        credit_column: 信用利差列名
        theme: 'light' 或 'dark'

    返回:
        plotly Figure 对象
    """
    config = get_theme_config(theme)

    if credit_df is None or credit_column not in credit_df.columns:
        # 如果没有信用利差数据，返回占位图
        fig = go.Figure()
        fig.add_annotation(
            text="信用利差数据未配置<br>请在 scripts/download_data.py 中添加 Wind EDB 指标",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=config['font_color'])
        )
        fig.update_layout(
            title='信用利差溢价分析',
            template=config['template'],
            height=400,
            paper_bgcolor=config['paper_bgcolor'],
            plot_bgcolor=config['plot_bgcolor'],
            font=dict(color=config['font_color'])
        )
        return fig

    # 合并数据计算溢价
    merged = pd.DataFrame({
        'local_gov': local_gov_df['spread_all'],
        'credit': credit_df[credit_column]
    }).dropna()

    merged['premium'] = merged['credit'] - merged['local_gov']

    fig = go.Figure()

    # 信用溢价面积图
    fig.add_trace(go.Scatter(
        x=merged.index,
        y=merged['premium'],
        mode='lines',
        name='信用溢价',
        fill='tozeroy',
        fillcolor='rgba(245, 158, 11, 0.3)',
        line=dict(color='#f59e0b', width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>溢价: %{y:.4f}<extra></extra>'
    ))

    # 溢价均值线
    avg_premium = merged['premium'].mean()
    fig.add_hline(
        y=avg_premium,
        line_dash='dash',
        line_color='#ef4444',
        line_width=2,
        annotation_text=f'平均溢价: {avg_premium:.4f}',
        annotation_position='right',
        annotation_font=dict(color='#ef4444', size=11)
    )

    fig.update_layout(
        title='信用利差溢价分析 (企业债 - 地方债)',
        xaxis_title='日期',
        yaxis_title='信用溢价',
        hovermode='x unified',
        template=config['template'],
        height=400,
        paper_bgcolor=config['paper_bgcolor'],
        plot_bgcolor=config['plot_bgcolor'],
        font=dict(color=config['font_color'])
    )

    fig.update_xaxes(gridcolor=config['grid_color'], linecolor=config['line_color'])
    fig.update_yaxes(gridcolor=config['grid_color'], linecolor=config['line_color'])

    # 添加区间选择器
    fig = add_range_selector(fig, theme == 'dark')

    return fig
