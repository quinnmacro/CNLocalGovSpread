"""
风险预警模块 - 风险监控和预警系统

功能:
1. 实时风险监控
2. 预警阈值配置
3. 预警历史记录
4. 风险仪表板
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================================
# 风险检查函数
# ============================================================================

def check_risk_alerts(clean_data, returns, evt, vol_modeler,
                      var_threshold=0.05, vol_percentile=0.95, deviation_threshold=1.5):
    """
    检查风险预警

    参数:
        clean_data: 清洗后的数据
        returns: 收益率序列
        evt: EVT分析器
        vol_modeler: 波动率建模器
        var_threshold: VaR预警阈值（绝对值）
        vol_percentile: 波动率预警百分位
        deviation_threshold: 偏离度预警阈值

    返回:
        list: 预警列表
    """
    alerts = []

    # 1. VaR 预警
    if evt is not None and evt.var is not None:
        var = evt.var
        current_return = returns.iloc[-1]
        var_breach = current_return > var

        if var_breach:
            alerts.append({
                'level': 'danger',
                'type': 'VaR超限',
                'message': f'当前收益 {current_return:.4f} 超过 99% VaR {var:.4f}，风险超限！',
                'value': current_return,
                'threshold': var,
                'timestamp': datetime.now()
            })
        elif current_return > var * 0.8:
            alerts.append({
                'level': 'warning',
                'type': 'VaR接近',
                'message': f'当前收益 {current_return:.4f} 接近 99% VaR {var:.4f}，风险较高',
                'value': current_return,
                'threshold': var,
                'timestamp': datetime.now()
            })
        else:
            alerts.append({
                'level': 'success',
                'type': 'VaR正常',
                'message': f'当前收益在 VaR 限额内，风险可控',
                'value': current_return,
                'threshold': var,
                'timestamp': datetime.now()
            })

    # 2. 波动率预警
    if vol_modeler is not None:
        try:
            winner = vol_modeler.run_tournament()
            winner_vol = vol_modeler.get_conditional_volatility(winner)
            current_vol = winner_vol.iloc[-1]
            vol_threshold_value = winner_vol.quantile(vol_percentile)

            if current_vol > vol_threshold_value:
                alerts.append({
                    'level': 'danger',
                    'type': '高波动预警',
                    'message': f'当前波动率 {current_vol:.4f} 超过 {vol_percentile*100:.0f}%分位 {vol_threshold_value:.4f}，市场波动剧烈！',
                    'value': current_vol,
                    'threshold': vol_threshold_value,
                    'timestamp': datetime.now()
                })
            elif current_vol > vol_threshold_value * 0.8:
                alerts.append({
                    'level': 'warning',
                    'type': '波动率上升',
                    'message': f'当前波动率 {current_vol:.4f} 接近预警阈值',
                    'value': current_vol,
                    'threshold': vol_threshold_value,
                    'timestamp': datetime.now()
                })
            else:
                alerts.append({
                    'level': 'success',
                    'type': '波动率正常',
                    'message': f'当前波动率 {current_vol:.4f} 在正常范围内',
                    'value': current_vol,
                    'threshold': vol_threshold_value,
                    'timestamp': datetime.now()
                })
        except:
            pass

    # 3. 利差异常预警
    if clean_data is not None:
        spread = clean_data['spread']
        current_spread = spread.iloc[-1]
        spread_mean = spread.mean()
        spread_std = spread.std()
        z_score = (current_spread - spread_mean) / spread_std

        if abs(z_score) > deviation_threshold * 1.5:
            level = 'danger'
            msg = f'利差 {current_spread:.2f} 偏离均值 {z_score:.1f}σ，异常波动！'
        elif abs(z_score) > deviation_threshold:
            level = 'warning'
            msg = f'利差 {current_spread:.2f} 偏离均值 {z_score:.1f}σ，需关注'
        else:
            level = 'success'
            msg = f'利差 {current_spread:.2f} 在正常范围内 (Z={z_score:.2f})'

        alerts.append({
            'level': level,
            'type': '利差异常',
            'message': msg,
            'value': current_spread,
            'threshold': spread_mean + deviation_threshold * spread_std,
            'z_score': z_score,
            'timestamp': datetime.now()
        })

    # 4. 趋势预警
    if len(spread) >= 20:
        recent_trend = spread.iloc[-20:].mean() - spread.iloc[-60:-20].mean()
        if recent_trend > spread_std * 0.5:
            alerts.append({
                'level': 'warning',
                'type': '上升趋势',
                'message': f'利差近20日均值上升 {recent_trend:.2f}，呈扩大趋势',
                'value': recent_trend,
                'timestamp': datetime.now()
            })
        elif recent_trend < -spread_std * 0.5:
            alerts.append({
                'level': 'warning',
                'type': '下降趋势',
                'message': f'利差近20日均值下降 {abs(recent_trend):.2f}，呈收窄趋势',
                'value': recent_trend,
                'timestamp': datetime.now()
            })

    return alerts


def get_risk_score(alerts):
    """
    计算综合风险评分

    参数:
        alerts: 预警列表

    返回:
        dict: 风险评分
    """
    score = 0
    danger_count = 0
    warning_count = 0

    for alert in alerts:
        if alert['level'] == 'danger':
            score += 3
            danger_count += 1
        elif alert['level'] == 'warning':
            score += 1
            warning_count += 1
        else:
            score += 0

    # 归一化到 0-100
    max_score = len(alerts) * 3
    normalized_score = (score / max_score * 100) if max_score > 0 else 0

    # 风险等级
    if normalized_score < 20:
        level = '低风险'
        color = 'green'
    elif normalized_score < 50:
        level = '中等风险'
        color = 'yellow'
    elif normalized_score < 75:
        level = '较高风险'
        color = 'orange'
    else:
        level = '高风险'
        color = 'red'

    return {
        'score': normalized_score,
        'level': level,
        'color': color,
        'danger_count': danger_count,
        'warning_count': warning_count,
        'total_alerts': len(alerts)
    }


# ============================================================================
# 预警历史
# ============================================================================

def generate_alert_history(clean_data, returns, window=252):
    """
    生成历史预警记录

    参数:
        clean_data: 清洗后的数据
        returns: 收益率序列
        window: 分析窗口

    返回:
        DataFrame: 历史预警记录
    """
    spread = clean_data['spread']
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()

    # 计算历史 Z-Score
    z_scores = (spread - spread_mean) / spread_std

    # 检测历史预警点
    alerts = []

    for i in range(window, len(spread)):
        z = z_scores.iloc[i]
        date = spread.index[i]
        value = spread.iloc[i]

        if abs(z) > 2.5:
            alerts.append({
                '日期': date.strftime('%Y-%m-%d'),
                '利差': value,
                'Z-Score': z,
                '预警级别': '危险' if abs(z) > 3 else '警告',
                '类型': '高估' if z > 0 else '低估'
            })

    if alerts:
        df = pd.DataFrame(alerts)
        return df.sort_values('日期', ascending=False).head(30)
    else:
        return pd.DataFrame(columns=['日期', '利差', 'Z-Score', '预警级别', '类型'])


def plot_alert_timeline(alert_history, theme='light'):
    """
    绘制预警时间线图

    参数:
        alert_history: 历史预警记录
        theme: 'light' 或 'dark'

    返回:
        plotly Figure 对象
    """
    template = 'plotly_dark' if theme == 'dark' else 'plotly_white'

    if alert_history.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="暂无历史预警记录",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(template=template, height=300)
        return fig

    # 转换日期
    dates = pd.to_datetime(alert_history['日期'])
    z_scores = alert_history['Z-Score'].abs()

    # 按级别着色
    colors = ['#ef4444' if level == '危险' else '#f59e0b'
              for level in alert_history['预警级别']]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=z_scores,
        mode='markers',
        marker=dict(
            size=12,
            color=colors,
            symbol=['triangle-up' if t == '高估' else 'triangle-down'
                    for t in alert_history['类型']]
        ),
        text=[f"{row['日期']}<br>利差: {row['利差']:.2f}<br>Z: {row['Z-Score']:.2f}"
              for _, row in alert_history.iterrows()],
        hoverinfo='text',
        name='预警点'
    ))

    # 添加阈值线
    fig.add_hline(y=2.5, line_dash='dash', line_color='#f59e0b',
                  annotation_text='警告阈值')
    fig.add_hline(y=3.0, line_dash='dash', line_color='#ef4444',
                  annotation_text='危险阈值')

    fig.update_layout(
        title='历史预警时间线',
        xaxis_title='日期',
        yaxis_title='|Z-Score|',
        template=template,
        height=400,
        showlegend=False
    )

    return fig


# ============================================================================
# 风险仪表板
# ============================================================================

def plot_risk_gauge(risk_score, theme='light'):
    """
    绘制风险仪表盘

    参数:
        risk_score: get_risk_score 的输出
        theme: 'light' 或 'dark'

    返回:
        plotly Figure 对象
    """
    template = 'plotly_dark' if theme == 'dark' else 'plotly_white'

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score['score'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"风险评分: {risk_score['level']}", 'font': {'size': 20}},
        delta={'reference': 30, 'increasing': {'color': "#ef4444"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': risk_score['color']},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#22c55e'},
                {'range': [20, 50], 'color': '#fbbf24'},
                {'range': [50, 75], 'color': '#f97316'},
                {'range': [75, 100], 'color': '#ef4444'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': risk_score['score']
            }
        }
    ))

    fig.update_layout(
        template=template,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def plot_risk_summary(alerts, theme='light'):
    """
    绘制风险汇总图

    参数:
        alerts: 预警列表
        theme: 'light' 或 'dark'

    返回:
        plotly Figure 对象
    """
    template = 'plotly_dark' if theme == 'dark' else 'plotly_white'

    # 统计各级别数量
    danger_count = sum(1 for a in alerts if a['level'] == 'danger')
    warning_count = sum(1 for a in alerts if a['level'] == 'warning')
    success_count = sum(1 for a in alerts if a['level'] == 'success')

    fig = go.Figure()

    categories = ['危险', '警告', '正常']
    values = [danger_count, warning_count, success_count]
    colors = ['#ef4444', '#f59e0b', '#22c55e']

    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=values,
        textposition='auto'
    ))

    fig.update_layout(
        title='预警统计',
        xaxis_title='预警级别',
        yaxis_title='数量',
        template=template,
        height=300,
        showlegend=False
    )

    return fig


# ============================================================================
# 预警配置
# ============================================================================

def get_default_thresholds():
    """获取默认预警阈值"""
    return {
        'var_threshold': 0.05,
        'vol_percentile': 0.95,
        'deviation_threshold': 1.5,
        'trend_threshold': 0.5
    }


def validate_thresholds(thresholds):
    """验证预警阈值"""
    default = get_default_thresholds()

    validated = {}
    for key, default_value in default.items():
        if key in thresholds:
            validated[key] = thresholds[key]
        else:
            validated[key] = default_value

    return validated


# ============================================================================
# 预警通知
# ============================================================================

def format_alert_message(alert):
    """
    格式化预警消息

    参数:
        alert: 预警字典

    返回:
        str: 格式化后的消息
    """
    level_icons = {
        'danger': '🔴',
        'warning': '🟡',
        'success': '🟢'
    }

    icon = level_icons.get(alert['level'], '⚪')
    msg = f"{icon} **{alert['type']}**: {alert['message']}"

    if 'value' in alert and 'threshold' in alert:
        msg += f"\n   - 当前值: {alert['value']:.4f}"
        msg += f"\n   - 阈值: {alert['threshold']:.4f}"

    return msg


def get_alert_summary(alerts):
    """
    获取预警摘要

    参数:
        alerts: 预警列表

    返回:
        str: 预警摘要文本
    """
    if not alerts:
        return "暂无预警信息"

    danger = [a for a in alerts if a['level'] == 'danger']
    warning = [a for a in alerts if a['level'] == 'warning']
    success = [a for a in alerts if a['level'] == 'success']

    summary = f"预警摘要: "
    summary += f"🔴 危险 {len(danger)} | "
    summary += f"🟡 警告 {len(warning)} | "
    summary += f"🟢 正常 {len(success)}"

    return summary
