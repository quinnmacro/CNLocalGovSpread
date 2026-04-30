"""
风险预警模块单元测试 - 测试 alerts.py 的所有核心功能
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from alerts import (
    check_risk_alerts,
    get_risk_score,
    generate_alert_history,
    plot_alert_timeline,
    plot_risk_gauge,
    plot_risk_summary,
    get_default_thresholds,
    validate_thresholds,
    format_alert_message,
    get_alert_summary
)
from evt import EVTRiskAnalyzer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """生成测试用利差数据"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=300, freq='B')
    spread = 80 + np.cumsum(np.random.randn(300) * 0.3)
    return pd.DataFrame({'spread': spread}, index=dates)


@pytest.fixture
def sample_returns():
    """生成测试用收益率数据（带肥尾）"""
    np.random.seed(42)
    return pd.Series(np.random.standard_t(5, 300) * 0.5)


@pytest.fixture
def sample_evt(sample_returns):
    """生成测试用 EVT 分析器"""
    evt = EVTRiskAnalyzer(sample_returns)
    evt.fit_gpd()
    evt.calculate_var()
    return evt


@pytest.fixture
def sample_alerts():
    """生成测试用预警列表"""
    return [
        {'level': 'danger', 'type': 'VaR超限', 'message': '风险超限',
         'value': 0.5, 'threshold': 0.3, 'timestamp': datetime.now()},
        {'level': 'warning', 'type': '波动率上升', 'message': '波动率较高',
         'value': 0.2, 'threshold': 0.15, 'timestamp': datetime.now()},
        {'level': 'success', 'type': 'VaR正常', 'message': '风险可控',
         'value': 0.1, 'threshold': 0.3, 'timestamp': datetime.now()},
        {'level': 'danger', 'type': '利差异常', 'message': '利差异常波动',
         'value': 120, 'threshold': 110, 'z_score': 2.5, 'timestamp': datetime.now()},
    ]


# ============================================================================
# check_risk_alerts 测试
# ============================================================================

class TestCheckRiskAlerts:

    def test_returns_list(self, sample_data, sample_returns, sample_evt):
        """测试 check_risk_alerts 返回列表"""
        alerts = check_risk_alerts(sample_data, sample_returns, sample_evt, None)
        assert isinstance(alerts, list)
        assert len(alerts) > 0

    def test_alert_has_required_fields(self, sample_data, sample_returns, sample_evt):
        """测试预警包含必要字段"""
        alerts = check_risk_alerts(sample_data, sample_returns, sample_evt, None)
        for alert in alerts:
            assert 'level' in alert
            assert 'type' in alert
            assert 'message' in alert
            assert 'timestamp' in alert

    def test_alert_levels_valid(self, sample_data, sample_returns, sample_evt):
        """测试预警级别有效"""
        alerts = check_risk_alerts(sample_data, sample_returns, sample_evt, None)
        valid_levels = {'danger', 'warning', 'success'}
        for alert in alerts:
            assert alert['level'] in valid_levels

    def test_var_alert_present(self, sample_data, sample_returns, sample_evt):
        """测试 VaR 预警存在"""
        alerts = check_risk_alerts(sample_data, sample_returns, sample_evt, None)
        var_alerts = [a for a in alerts if a['type'] in ('VaR超限', 'VaR接近', 'VaR正常')]
        assert len(var_alerts) > 0

    def test_spread_alert_present(self, sample_data, sample_returns, sample_evt):
        """测试利差异常预警存在"""
        alerts = check_risk_alerts(sample_data, sample_returns, sample_evt, None)
        spread_alerts = [a for a in alerts if a['type'] == '利差异常']
        assert len(spread_alerts) > 0

    def test_none_evt_graceful(self, sample_data, sample_returns):
        """测试 EVT 为 None 时优雅处理"""
        alerts = check_risk_alerts(sample_data, sample_returns, None, None)
        assert isinstance(alerts, list)
        var_alerts = [a for a in alerts if 'VaR' in a['type']]
        assert len(var_alerts) == 0

    def test_trend_alert_with_long_data(self, sample_data, sample_returns, sample_evt):
        """测试趋势预警（需要>=60天数据）"""
        alerts = check_risk_alerts(sample_data, sample_returns, sample_evt, None)
        trend_alerts = [a for a in alerts if a['type'] in ('上升趋势', '下降趋势')]
        assert isinstance(trend_alerts, list)

    def test_custom_thresholds(self, sample_data, sample_returns, sample_evt):
        """测试自定义阈值"""
        alerts = check_risk_alerts(
            sample_data, sample_returns, sample_evt, None,
            var_threshold=0.01, vol_percentile=0.90, deviation_threshold=1.0
        )
        assert isinstance(alerts, list)
        assert len(alerts) > 0


# ============================================================================
# get_risk_score 测试
# ============================================================================

class TestGetRiskScore:

    def test_returns_dict(self, sample_alerts):
        """测试 get_risk_score 返回字典"""
        score = get_risk_score(sample_alerts)
        assert isinstance(score, dict)

    def test_has_required_fields(self, sample_alerts):
        """测试包含必要字段"""
        score = get_risk_score(sample_alerts)
        assert 'score' in score
        assert 'level' in score
        assert 'color' in score
        assert 'danger_count' in score
        assert 'warning_count' in score
        assert 'total_alerts' in score

    def test_danger_count_correct(self, sample_alerts):
        """测试危险计数正确"""
        score = get_risk_score(sample_alerts)
        expected_danger = sum(1 for a in sample_alerts if a['level'] == 'danger')
        assert score['danger_count'] == expected_danger

    def test_warning_count_correct(self, sample_alerts):
        """测试警告计数正确"""
        score = get_risk_score(sample_alerts)
        expected_warning = sum(1 for a in sample_alerts if a['level'] == 'warning')
        assert score['warning_count'] == expected_warning

    def test_score_range(self, sample_alerts):
        """测试评分在合理范围"""
        score = get_risk_score(sample_alerts)
        assert 0 <= score['score'] <= 100

    def test_empty_alerts(self):
        """测试空预警列表"""
        score = get_risk_score([])
        assert score['score'] == 0
        assert score['level'] == '低风险'
        assert score['color'] == 'green'

    def test_all_danger(self):
        """测试全部危险预警"""
        danger_alerts = [
            {'level': 'danger', 'type': 'VaR超限', 'message': '危险',
             'value': 0.5, 'threshold': 0.3, 'timestamp': datetime.now()}
        ] * 3
        score = get_risk_score(danger_alerts)
        assert score['score'] == 100
        assert score['level'] == '高风险'
        assert score['color'] == 'red'

    def test_all_success(self):
        """测试全部正常预警"""
        success_alerts = [
            {'level': 'success', 'type': 'VaR正常', 'message': '正常',
             'value': 0.1, 'threshold': 0.3, 'timestamp': datetime.now()}
        ] * 3
        score = get_risk_score(success_alerts)
        assert score['score'] == 0
        assert score['level'] == '低风险'


# ============================================================================
# generate_alert_history 测试
# ============================================================================

class TestGenerateAlertHistory:

    def test_returns_dataframe(self, sample_data, sample_returns):
        """测试返回 DataFrame"""
        history = generate_alert_history(sample_data, sample_returns)
        assert isinstance(history, pd.DataFrame)

    def test_has_expected_columns(self, sample_data, sample_returns):
        """测试包含预期列"""
        history = generate_alert_history(sample_data, sample_returns)
        expected_cols = ['日期', '利差', 'Z-Score', '预警级别', '类型']
        for col in expected_cols:
            assert col in history.columns

    def test_max_30_rows(self, sample_data, sample_returns):
        """测试最多30行"""
        history = generate_alert_history(sample_data, sample_returns)
        assert len(history) <= 30

    def test_custom_window(self, sample_data, sample_returns):
        """测试自定义窗口"""
        history = generate_alert_history(sample_data, sample_returns, window=120)
        assert isinstance(history, pd.DataFrame)


# ============================================================================
# format_alert_message 测试
# ============================================================================

class TestFormatAlertMessage:

    def test_danger_format(self):
        """测试危险消息格式"""
        alert = {'level': 'danger', 'type': 'VaR超限', 'message': '风险超限',
                 'value': 0.5, 'threshold': 0.3}
        msg = format_alert_message(alert)
        assert 'VaR超限' in msg
        assert '0.5000' in msg
        assert '0.3000' in msg

    def test_warning_format(self):
        """测试警告消息格式"""
        alert = {'level': 'warning', 'type': '波动率上升', 'message': '波动率较高'}
        msg = format_alert_message(alert)
        assert '波动率上升' in msg

    def test_success_format(self):
        """测试正常消息格式"""
        alert = {'level': 'success', 'type': 'VaR正常', 'message': '风险可控'}
        msg = format_alert_message(alert)
        assert 'VaR正常' in msg


# ============================================================================
# get_alert_summary 测试
# ============================================================================

class TestGetAlertSummary:

    def test_summary_format(self, sample_alerts):
        """测试摘要格式"""
        summary = get_alert_summary(sample_alerts)
        assert '危险' in summary
        assert '警告' in summary
        assert '正常' in summary

    def test_empty_summary(self):
        """测试空预警摘要"""
        summary = get_alert_summary([])
        assert '暂无' in summary

    def test_counts_correct(self, sample_alerts):
        """测试计数正确"""
        summary = get_alert_summary(sample_alerts)
        danger_count = sum(1 for a in sample_alerts if a['level'] == 'danger')
        assert str(danger_count) in summary


# ============================================================================
# 预警阈值配置测试
# ============================================================================

class TestThresholds:

    def test_default_thresholds(self):
        """测试默认阈值"""
        thresholds = get_default_thresholds()
        assert 'var_threshold' in thresholds
        assert 'vol_percentile' in thresholds
        assert 'deviation_threshold' in thresholds
        assert 'trend_threshold' in thresholds
        assert thresholds['var_threshold'] == 0.05
        assert thresholds['vol_percentile'] == 0.95

    def test_validate_partial_thresholds(self):
        """测试部分阈值验证"""
        custom = {'var_threshold': 0.01}
        validated = validate_thresholds(custom)
        assert validated['var_threshold'] == 0.01
        assert validated['vol_percentile'] == 0.95  # 默认值

    def test_validate_complete_thresholds(self):
        """测试完整阈值验证"""
        custom = {'var_threshold': 0.01, 'vol_percentile': 0.90,
                  'deviation_threshold': 1.0, 'trend_threshold': 0.3}
        validated = validate_thresholds(custom)
        assert validated == custom

    def test_validate_empty_thresholds(self):
        """测试空阈值验证"""
        validated = validate_thresholds({})
        defaults = get_default_thresholds()
        assert validated == defaults


# ============================================================================
# 可视化测试
# ============================================================================

class TestVisualization:

    def test_alert_timeline_returns_figure(self, sample_data, sample_returns):
        """测试预警时间线返回 Figure"""
        history = generate_alert_history(sample_data, sample_returns)
        fig = plot_alert_timeline(history)
        assert isinstance(fig, go.Figure)

    def test_alert_timeline_empty(self):
        """测试空历史记录时间线"""
        empty_history = pd.DataFrame(columns=['日期', '利差', 'Z-Score', '预警级别', '类型'])
        fig = plot_alert_timeline(empty_history)
        assert isinstance(fig, go.Figure)

    def test_alert_timeline_dark_theme(self, sample_data, sample_returns):
        """测试深色主题时间线"""
        history = generate_alert_history(sample_data, sample_returns)
        fig = plot_alert_timeline(history, theme='dark')
        assert isinstance(fig, go.Figure)

    def test_risk_gauge_returns_figure(self, sample_alerts):
        """测试风险仪表盘返回 Figure"""
        score = get_risk_score(sample_alerts)
        fig = plot_risk_gauge(score)
        assert isinstance(fig, go.Figure)

    def test_risk_gauge_dark_theme(self, sample_alerts):
        """测试深色主题仪表盘"""
        score = get_risk_score(sample_alerts)
        fig = plot_risk_gauge(score, theme='dark')
        assert isinstance(fig, go.Figure)

    def test_risk_summary_returns_figure(self, sample_alerts):
        """测试风险汇总图返回 Figure"""
        fig = plot_risk_summary(sample_alerts)
        assert isinstance(fig, go.Figure)

    def test_risk_summary_dark_theme(self, sample_alerts):
        """测试深色主题汇总图"""
        fig = plot_risk_summary(sample_alerts, theme='dark')
        assert isinstance(fig, go.Figure)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
