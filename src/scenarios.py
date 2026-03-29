"""
情景分析模块 - 压力测试、蒙特卡洛模拟、敏感性分析

功能:
1. 压力测试 - 利差冲击下的风险分析
2. 蒙特卡洛模拟 - 未来分布预测
3. 敏感性分析 - 参数敏感性
4. 历史回溯分析 - 滚动统计和事件检测
"""

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================================
# 压力测试
# ============================================================================

def run_stress_test(returns, shock, confidence=0.99):
    """
    运行压力测试

    参数:
        returns: 收益率序列
        shock: 利差冲击值 (bps)
        confidence: VaR置信水平

    返回:
        dict: 压力测试结果
    """
    # 添加冲击后的收益率分布
    stressed_returns = returns + shock

    # 计算压力VaR
    var = stressed_returns.quantile(1 - confidence)

    # 计算压力ES
    es_threshold = stressed_returns.quantile(1 - confidence)
    es = stressed_returns[stressed_returns > es_threshold].mean() if len(stressed_returns[stressed_returns > es_threshold]) > 0 else es_threshold

    # 计算最大损失
    max_loss = stressed_returns.max()

    # 计算冲击影响
    original_var = returns.quantile(1 - confidence)
    var_change = var - original_var

    return {
        'var': abs(var),
        'es': abs(es),
        'max_loss': abs(max_loss),
        'original_var': abs(original_var),
        'var_change': abs(var_change),
        'shock': shock
    }


def run_multi_scenario_stress(returns, shock_range=(-50, 50), n_scenarios=21):
    """
    运行多情景压力测试

    参数:
        returns: 收益率序列
        shock_range: 冲击范围 (min, max)
        n_scenarios: 情景数量

    返回:
        DataFrame: 多情景压力测试结果
    """
    shocks = np.linspace(shock_range[0], shock_range[1], n_scenarios)
    results = []

    for shock in shocks:
        result = run_stress_test(returns, shock)
        results.append({
            'shock': shock,
            'var': result['var'],
            'es': result['es'],
            'max_loss': result['max_loss']
        })

    return pd.DataFrame(results)


# ============================================================================
# 蒙特卡洛模拟
# ============================================================================

def run_monte_carlo(returns, n_simulations=10000, horizon=10, seed=None):
    """
    运行蒙特卡洛模拟

    参数:
        returns: 历史收益率序列
        n_simulations: 模拟次数
        horizon: 预测天数
        seed: 随机种子

    返回:
        dict: 模拟结果
    """
    if seed is not None:
        np.random.seed(seed)

    # 估计参数
    mu = returns.mean()
    sigma = returns.std()

    # 使用 t 分布拟合尾部
    df, loc, scale = stats.t.fit(returns)

    # 模拟路径
    n_steps = horizon
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = 0  # 初始值

    # 生成随机数（使用 t 分布）
    for t in range(1, n_steps + 1):
        random_shocks = stats.t.rvs(df, loc=loc, scale=scale, size=n_simulations)
        paths[:, t] = paths[:, t - 1] + random_shocks

    # 计算最终分布
    final_values = paths[:, -1]

    # 计算统计量
    results = {
        'paths': paths,
        'final_values': final_values,
        'mean': final_values.mean(),
        'std': final_values.std(),
        'var_95': np.percentile(final_values, 95),
        'var_99': np.percentile(final_values, 99),
        'es_99': final_values[final_values >= np.percentile(final_values, 99)].mean(),
        'min': final_values.min(),
        'max': final_values.max(),
        'median': np.median(final_values),
        'params': {
            'mu': mu,
            'sigma': sigma,
            'df': df,
            'loc': loc,
            'scale': scale
        }
    }

    return results


def plot_mc_simulation(mc_results, theme='light'):
    """
    绘制蒙特卡洛模拟结果

    参数:
        mc_results: run_monte_carlo 的输出
        theme: 'light' 或 'dark'

    返回:
        plotly Figure 对象
    """
    template = 'plotly_dark' if theme == 'dark' else 'plotly_white'
    final_values = mc_results['final_values']

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('最终分布', '风险指标'),
        column_widths=[0.6, 0.4]
    )

    # 直方图
    fig.add_trace(
        go.Histogram(
            x=final_values,
            nbinsx=50,
            name='模拟结果',
            marker_color='#3b82f6',
            opacity=0.7
        ),
        row=1, col=1
    )

    # VaR 线
    fig.add_vline(
        x=mc_results['var_99'],
        line_dash='dash',
        line_color='#ef4444',
        annotation_text=f"99% VaR: {mc_results['var_99']:.4f}",
        annotation_position='top',
        row=1, col=1
    )

    # 风险指标条形图
    metrics = ['VaR 95%', 'VaR 99%', 'ES 99%']
    values = [mc_results['var_95'], mc_results['var_99'], mc_results['es_99']]
    colors = ['#f59e0b', '#ef4444', '#dc2626']

    fig.add_trace(
        go.Bar(
            x=metrics,
            y=values,
            marker_color=colors,
            name='风险指标'
        ),
        row=1, col=2
    )

    fig.update_layout(
        template=template,
        height=400,
        showlegend=False
    )

    return fig


def plot_mc_paths(mc_results, n_paths=50, theme='light'):
    """
    绘制蒙特卡洛模拟路径

    参数:
        mc_results: run_monte_carlo 的输出
        n_paths: 显示的路径数
        theme: 'light' 或 'dark'

    返回:
        plotly Figure 对象
    """
    template = 'plotly_dark' if theme == 'dark' else 'plotly_white'
    paths = mc_results['paths'][:n_paths]
    horizon = paths.shape[1] - 1

    fig = go.Figure()

    # 绘制路径
    for i in range(min(n_paths, len(paths))):
        fig.add_trace(go.Scatter(
            x=list(range(horizon + 1)),
            y=paths[i],
            mode='lines',
            line=dict(color='#3b82f6', width=0.5),
            opacity=0.3,
            showlegend=False
        ))

    # 添加均值线
    mean_path = mc_results['paths'].mean(axis=0)
    fig.add_trace(go.Scatter(
        x=list(range(horizon + 1)),
        y=mean_path,
        mode='lines',
        line=dict(color='#ef4444', width=2),
        name='均值路径'
    ))

    # 添加置信区间
    lower_95 = np.percentile(mc_results['paths'], 2.5, axis=0)
    upper_95 = np.percentile(mc_results['paths'], 97.5, axis=0)

    fig.add_trace(go.Scatter(
        x=list(range(horizon + 1)),
        y=upper_95,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=list(range(horizon + 1)),
        y=lower_95,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.2)',
        name='95% 置信区间'
    ))

    fig.update_layout(
        title='蒙特卡洛模拟路径',
        xaxis_title='天数',
        yaxis_title='累计变化',
        template=template,
        height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig


# ============================================================================
# 敏感性分析
# ============================================================================

def run_sensitivity_analysis(returns, param='volatility', n_points=20):
    """
    运行敏感性分析

    参数:
        returns: 收益率序列
        param: 分析参数 ('volatility', 'mean', 'df')
        n_points: 分析点数

    返回:
        DataFrame: 敏感性分析结果
    """
    base_mu = returns.mean()
    base_sigma = returns.std()
    df, loc, scale = stats.t.fit(returns)

    results = []

    if param == 'volatility':
        # 波动率敏感性
        sigma_range = np.linspace(base_sigma * 0.5, base_sigma * 2, n_points)
        for sigma in sigma_range:
            simulated = stats.t.rvs(df, loc=base_mu, scale=sigma, size=10000)
            var_99 = np.percentile(simulated, 99)
            es_99 = simulated[simulated >= var_99].mean()
            results.append({
                'parameter': 'volatility',
                'value': sigma,
                'var_99': var_99,
                'es_99': es_99
            })

    elif param == 'mean':
        # 均值敏感性
        mu_range = np.linspace(base_mu - base_sigma, base_mu + base_sigma, n_points)
        for mu in mu_range:
            simulated = stats.t.rvs(df, loc=mu, scale=base_sigma, size=10000)
            var_99 = np.percentile(simulated, 99)
            es_99 = simulated[simulated >= var_99].mean()
            results.append({
                'parameter': 'mean',
                'value': mu,
                'var_99': var_99,
                'es_99': es_99
            })

    elif param == 'df':
        # 自由度敏感性（尾部厚度）
        df_range = np.linspace(2, 30, n_points)
        for d in df_range:
            simulated = stats.t.rvs(d, loc=base_mu, scale=scale, size=10000)
            var_99 = np.percentile(simulated, 99)
            es_99 = simulated[simulated >= var_99].mean()
            results.append({
                'parameter': 'df',
                'value': d,
                'var_99': var_99,
                'es_99': es_99
            })

    return pd.DataFrame(results)


def plot_sensitivity_analysis(sensitivity_df, param='volatility', theme='light'):
    """
    绘制敏感性分析图

    参数:
        sensitivity_df: 敏感性分析结果 DataFrame
        param: 分析参数
        theme: 'light' 或 'dark'

    返回:
        plotly Figure 对象
    """
    template = 'plotly_dark' if theme == 'dark' else 'plotly_white'

    param_names = {
        'volatility': '波动率',
        'mean': '均值',
        'df': '自由度'
    }

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('VaR 敏感性', 'ES 敏感性')
    )

    # VaR 敏感性
    fig.add_trace(
        go.Scatter(
            x=sensitivity_df['value'],
            y=sensitivity_df['var_99'],
            mode='lines+markers',
            name='VaR 99%',
            line=dict(color='#3b82f6', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )

    # ES 敏感性
    fig.add_trace(
        go.Scatter(
            x=sensitivity_df['value'],
            y=sensitivity_df['es_99'],
            mode='lines+markers',
            name='ES 99%',
            line=dict(color='#ef4444', width=2),
            marker=dict(size=6)
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=f'{param_names.get(param, param)}敏感性分析',
        template=template,
        height=400,
        showlegend=False
    )

    fig.update_xaxes(title_text=param_names.get(param, param))
    fig.update_yaxes(title_text='VaR 99%', row=1, col=1)
    fig.update_yaxes(title_text='ES 99%', row=1, col=2)

    return fig


# ============================================================================
# 历史回溯分析
# ============================================================================

def calculate_rolling_stats(data, window=60):
    """
    计算滚动统计量

    参数:
        data: 数据序列
        window: 滚动窗口

    返回:
        DataFrame: 滚动统计量
    """
    df = pd.DataFrame(index=data.index)

    df['rolling_mean'] = data['spread'].rolling(window=window).mean()
    df['rolling_std'] = data['spread'].rolling(window=window).std()
    df['rolling_min'] = data['spread'].rolling(window=window).min()
    df['rolling_max'] = data['spread'].rolling(window=window).max()
    df['rolling_range'] = df['rolling_max'] - df['rolling_min']

    # 滚动分位数
    df['rolling_q25'] = data['spread'].rolling(window=window).quantile(0.25)
    df['rolling_q75'] = data['spread'].rolling(window=window).quantile(0.75)
    df['rolling_iqr'] = df['rolling_q75'] - df['rolling_q25']

    # 滚动偏度和峰度
    df['rolling_skew'] = data['spread'].rolling(window=window).skew()
    df['rolling_kurt'] = data['spread'].rolling(window=window).kurt()

    return df


def detect_historical_events(data, threshold=3.0):
    """
    检测历史重要事件

    参数:
        data: 数据序列
        threshold: 标准差阈值

    返回:
        DataFrame: 检测到的事件
    """
    spread = data['spread']
    mean = spread.mean()
    std = spread.std()

    # 检测异常值
    anomalies = spread[(spread - mean).abs() > threshold * std]

    # 检测大变化
    changes = spread.diff().abs()
    big_changes = changes[changes > changes.quantile(0.95)]

    events = []

    # 添加异常值事件
    for date, value in anomalies.items():
        z_score = (value - mean) / std
        events.append({
            '日期': date.strftime('%Y-%m-%d'),
            '类型': '异常值',
            '描述': f'利差 {value:.2f} (偏离 {z_score:.1f}σ)',
            '影响': '高' if abs(z_score) > 4 else '中'
        })

    # 添加大变化事件
    for date, change in big_changes.items():
        events.append({
            '日期': date.strftime('%Y-%m-%d'),
            '类型': '大幅波动',
            '描述': f'单日变化 {change:.2f}',
            '影响': '高' if change > changes.quantile(0.99) else '中'
        })

    return pd.DataFrame(events).sort_values('日期', ascending=False).head(20)


def plot_rolling_stats(rolling_stats, original_data=None, theme='light'):
    """
    绘制滚动统计图

    参数:
        rolling_stats: calculate_rolling_stats 的输出
        original_data: 原始数据（可选）
        theme: 'light' 或 'dark'

    返回:
        plotly Figure 对象
    """
    template = 'plotly_dark' if theme == 'dark' else 'plotly_white'

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('滚动均值±标准差', '滚动范围', '滚动偏度', '滚动峰度')
    )

    # 滚动均值和标准差
    fig.add_trace(
        go.Scatter(
            x=rolling_stats.index,
            y=rolling_stats['rolling_mean'],
            mode='lines',
            name='滚动均值',
            line=dict(color='#3b82f6', width=2)
        ),
        row=1, col=1
    )

    # 标准差带
    fig.add_trace(
        go.Scatter(
            x=rolling_stats.index,
            y=rolling_stats['rolling_mean'] + rolling_stats['rolling_std'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_stats.index,
            y=rolling_stats['rolling_mean'] - rolling_stats['rolling_std'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(59, 130, 246, 0.2)',
            name='±1σ'
        ),
        row=1, col=1
    )

    # 滚动范围
    fig.add_trace(
        go.Scatter(
            x=rolling_stats.index,
            y=rolling_stats['rolling_range'],
            mode='lines',
            name='滚动范围',
            line=dict(color='#22c55e', width=2),
            fill='tozeroy',
            fillcolor='rgba(34, 197, 94, 0.2)'
        ),
        row=1, col=2
    )

    # 滚动偏度
    fig.add_trace(
        go.Scatter(
            x=rolling_stats.index,
            y=rolling_stats['rolling_skew'],
            mode='lines',
            name='滚动偏度',
            line=dict(color='#f59e0b', width=2)
        ),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash='dash', line_color='gray', row=2, col=1)

    # 滚动峰度
    fig.add_trace(
        go.Scatter(
            x=rolling_stats.index,
            y=rolling_stats['rolling_kurt'],
            mode='lines',
            name='滚动峰度',
            line=dict(color='#ef4444', width=2)
        ),
        row=2, col=2
    )
    fig.add_hline(y=0, line_dash='dash', line_color='gray', row=2, col=2)

    fig.update_layout(
        template=template,
        height=600,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig


def plot_percentile_chart(data, windows=[20, 60, 120], theme='light'):
    """
    绘制历史分位数图

    参数:
        data: 数据序列
        windows: 计算窗口列表
        theme: 'light' 或 'dark'

    返回:
        plotly Figure 对象
    """
    template = 'plotly_dark' if theme == 'dark' else 'plotly_white'

    fig = go.Figure()

    spread = data['spread']

    # 当前值
    current = spread.iloc[-1]

    # 各窗口分位数
    colors = ['#3b82f6', '#22c55e', '#f59e0b']

    for i, window in enumerate(windows):
        recent = spread.iloc[-window:]
        q25 = recent.quantile(0.25)
        q50 = recent.quantile(0.50)
        q75 = recent.quantile(0.75)

        # 计算当前值在分位数中的位置
        percentile = (recent < current).mean() * 100

        fig.add_trace(go.Bar(
            x=[f'{window}日'],
            y=[q75 - q25],
            base=q25,
            marker_color=colors[i],
            opacity=0.5,
            name=f'{window}日 IQR',
            showlegend=True
        ))

        # 标记中位数
        fig.add_trace(go.Scatter(
            x=[f'{window}日'],
            y=[q50],
            mode='markers',
            marker=dict(symbol='diamond', size=10, color=colors[i]),
            name=f'{window}日中位数',
            showlegend=False
        ))

    # 当前值线
    fig.add_hline(
        y=current,
        line_dash='dash',
        line_color='#ef4444',
        annotation_text=f'当前: {current:.2f}'
    )

    fig.update_layout(
        title='历史分位数分析',
        yaxis_title='利差',
        template=template,
        height=400,
        barmode='overlay',
        showlegend=True
    )

    return fig
