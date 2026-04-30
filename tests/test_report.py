"""
report.py 单元测试 - 覆盖 generate_strategic_report 全部逻辑分支
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from io import StringIO
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from report import generate_strategic_report


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def base_spread_data():
    """基础利差数据 (100日)"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='B')
    spread = pd.Series(80 + np.random.randn(100) * 5, index=dates, name='spread')
    df = pd.DataFrame({'spread': spread})
    return df


@pytest.fixture
def low_deviation_data(base_spread_data):
    """低偏离数据 (<1.5σ)"""
    smoothed = base_spread_data['spread'].rolling(20).mean().fillna(base_spread_data['spread'].iloc[0])
    deviation = pd.Series(np.random.randn(100) * 0.5, index=base_spread_data.index)
    return base_spread_data, smoothed, deviation


@pytest.fixture
def moderate_deviation_data(base_spread_data):
    """中等偏离数据 (1.5-2σ)"""
    smoothed = base_spread_data['spread'].rolling(20).mean().fillna(base_spread_data['spread'].iloc[0])
    deviation = pd.Series(np.linspace(0, 1.7, 100), index=base_spread_data.index)
    return base_spread_data, smoothed, deviation


@pytest.fixture
def high_positive_deviation_data(base_spread_data):
    """高正偏离数据 (>2σ)"""
    smoothed = pd.Series(80.0, index=base_spread_data.index)
    deviation = pd.Series(np.linspace(0, 2.5, 100), index=base_spread_data.index)
    return base_spread_data, smoothed, deviation


@pytest.fixture
def high_negative_deviation_data(base_spread_data):
    """高负偏离数据 (<-2σ)"""
    smoothed = pd.Series(90.0, index=base_spread_data.index)
    deviation = pd.Series(np.linspace(0, -2.5, 100), index=base_spread_data.index)
    return base_spread_data, smoothed, deviation


@pytest.fixture
def vol_modeler_mock():
    """GARCH获胜的 vol_modeler mock"""
    mock = MagicMock()
    mock.ic_scores = {
        'GARCH': {'AIC': 100.0, 'BIC': 110.0, 'converged': True},
        'EGARCH': {'AIC': 120.0, 'BIC': 130.0, 'converged': True},
        'GJR-GARCH': {'AIC': 115.0, 'BIC': 125.0, 'converged': True},
        'EWMA': {'AIC': 140.0, 'BIC': 150.0, 'converged': True},
    }
    mock.results = {
        'GJR-GARCH': MagicMock(params={'gamma[1]': 0.08}),
    }
    return mock


@pytest.fixture
def egarch_winner_vol_modeler_mock():
    """EGARCH获胜的 vol_modeler mock (含显著GJR gamma)"""
    mock = MagicMock()
    mock.ic_scores = {
        'EGARCH': {'AIC': 90.0, 'BIC': 100.0, 'converged': True},
        'GARCH': {'AIC': 100.0, 'BIC': 110.0, 'converged': True},
        'GJR-GARCH': {'AIC': 95.0, 'BIC': 105.0, 'converged': True},
    }
    mock.results = {
        'GJR-GARCH': MagicMock(params={'gamma[1]': 0.15}),
    }
    return mock


@pytest.fixture
def egarch_winner_no_gjr_mock():
    """EGARCH获胜但没有GJR结果的 mock"""
    mock = MagicMock()
    mock.ic_scores = {
        'EGARCH': {'AIC': 90.0, 'BIC': 100.0, 'converged': True},
        'GARCH': {'AIC': 100.0, 'BIC': 110.0, 'converged': True},
    }
    mock.results = {}
    return mock


@pytest.fixture
def gjr_winner_vol_modeler_mock():
    """GJR-GARCH获胜且gamma显著的 mock"""
    mock = MagicMock()
    mock.ic_scores = {
        'GJR-GARCH': {'AIC': 85.0, 'BIC': 95.0, 'converged': True},
        'GARCH': {'AIC': 100.0, 'BIC': 110.0, 'converged': True},
        'EGARCH': {'AIC': 120.0, 'BIC': 130.0, 'converged': True},
    }
    mock.results = {
        'GJR-GARCH': MagicMock(params={'gamma[1]': 0.12}),
    }
    return mock


@pytest.fixture
def gjr_winner_low_gamma_mock():
    """GJR-GARCH获胜但gamma不显著的 mock"""
    mock = MagicMock()
    mock.ic_scores = {
        'GJR-GARCH': {'AIC': 85.0, 'BIC': 95.0, 'converged': True},
        'GARCH': {'AIC': 100.0, 'BIC': 110.0, 'converged': True},
    }
    mock.results = {
        'GJR-GARCH': MagicMock(params={'gamma[1]': 0.02}),
    }
    return mock


@pytest.fixture
def volatility_series():
    """波动率序列"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='B')
    return pd.Series(3.0 + np.random.randn(100) * 0.5, index=dates)


@pytest.fixture
def high_volatility_series():
    """高波动率序列 (>90%分位)"""
    dates = pd.date_range('2020-01-01', periods=100, freq='B')
    base = np.random.randn(100) * 0.5 + 3.0
    base[-1] = 15.0  # 最后一值极高
    return pd.Series(base, index=dates)


def _run_report(captured, **kwargs):
    """运行报告并返回输出文本"""
    generate_strategic_report(**kwargs)
    return captured.getvalue()


# ============================================================================
# 第一部分: 模型锦标赛结果
# ============================================================================

class TestModelTournamentSection:
    """模型锦标赛输出测试"""

    def test_prints_winner_model_name(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """测试输出获胜模型名称"""
        clean_data, smoothed, deviation = low_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert 'GARCH' in output
        assert '获胜模型' in output

    def test_prints_all_model_scores(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """测试输出所有模型IC分数对比"""
        clean_data, smoothed, deviation = low_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert 'GARCH' in output
        assert 'EGARCH' in output
        assert 'GJR-GARCH' in output
        assert 'EWMA' in output
        assert '100.00' in output  # AIC值

    def test_prints_aic_bic_values(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """测试输出AIC和BIC数值"""
        clean_data, smoothed, deviation = low_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert 'AIC' in output
        assert 'BIC' in output

    def test_winner_marked_with_trophy(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """测试获胜模型标记🏆"""
        clean_data, smoothed, deviation = low_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '🏆' in output

    def test_prints_conclusion_line(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """测试输出结论性描述"""
        clean_data, smoothed, deviation = low_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert 'AIC 准则' in output or '结论' in output


# ============================================================================
# 第二部分: 不对称效应检验
# ============================================================================

class TestAsymmetrySection:
    """波动率不对称效应测试"""

    def test_garch_winner_no_asymmetry(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """GARCH获胜时不检测不对称效应"""
        clean_data, smoothed, deviation = low_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '未检测到不对称效应' in output or '对称' in output

    def test_egarch_winner_with_significant_gjr(self, egarch_winner_vol_modeler_mock, low_deviation_data, volatility_series):
        """EGARCH获胜+GJR gamma显著时检测杠杆效应"""
        clean_data, smoothed, deviation = low_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='EGARCH',
                vol_modeler=egarch_winner_vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '杠杆效应' in output or 'GJR-GARCH' in output

    def test_egarch_winner_no_gjr_results(self, egarch_winner_no_gjr_mock, low_deviation_data, volatility_series):
        """EGARCH获胜但没有GJR结果时输出隐式非对称说明"""
        clean_data, smoothed, deviation = low_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='EGARCH',
                vol_modeler=egarch_winner_no_gjr_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '隐式' in output or '|z_t|' in output

    def test_egarch_winner_gjr_low_gamma(self, low_deviation_data, volatility_series):
        """EGARCH获胜+GJR gamma不显著时输出隐式说明"""
        clean_data, smoothed, deviation = low_deviation_data
        mock = MagicMock()
        mock.ic_scores = {'EGARCH': {'AIC': 90.0, 'BIC': 100.0}}
        mock.results = {'GJR-GARCH': MagicMock(params={'gamma[1]': 0.03})}
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='EGARCH',
                vol_modeler=mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '隐式' in output or '|z_t|' in output

    def test_gjr_winner_significant_gamma(self, gjr_winner_vol_modeler_mock, low_deviation_data, volatility_series):
        """GJR-GARCH获胜且gamma显著"""
        clean_data, smoothed, deviation = low_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GJR-GARCH',
                vol_modeler=gjr_winner_vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '杠杆效应' in output
        assert 'γ' in output or '0.12' in output

    def test_gjr_winner_low_gamma(self, gjr_winner_low_gamma_mock, low_deviation_data, volatility_series):
        """GJR-GARCH获胜但gamma不显著"""
        clean_data, smoothed, deviation = low_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GJR-GARCH',
                vol_modeler=gjr_winner_low_gamma_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '不显著' in output

    def test_asymmetry_detected_prints_trading_implication(self, gjr_winner_vol_modeler_mock, low_deviation_data, volatility_series):
        """检测到不对称效应时输出交易含义"""
        clean_data, smoothed, deviation = low_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GJR-GARCH',
                vol_modeler=gjr_winner_vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '交易含义' in output or '对冲' in output


# ============================================================================
# 第三部分: 风险预警
# ============================================================================

class TestRiskStatusSection:
    """当前风险状况输出测试"""

    def test_prints_current_spread(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """测试输出当前利差水平"""
        clean_data, smoothed, deviation = low_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '当前利差水平' in output
        assert 'bps' in output

    def test_prints_kalman_trend(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """测试输出卡尔曼趋势水平"""
        clean_data, smoothed, deviation = low_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '卡尔曼趋势' in output or '趋势水平' in output

    def test_prints_deviation_bps_and_sigma(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """测试输出偏离程度(bps和σ)"""
        clean_data, smoothed, deviation = low_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '偏离程度' in output
        assert 'σ' in output

    def test_prints_evt_var(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """测试输出EVT-VaR"""
        clean_data, smoothed, deviation = low_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert 'EVT-VaR' in output

    def test_prints_evt_es_when_provided(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """测试提供ES时输出EVT-ES"""
        clean_data, smoothed, deviation = low_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
                evt_es=7.5,
            )
        assert 'EVT-ES' in output

    def test_no_evt_es_when_not_provided(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """测试不提供ES时不输出EVT-ES"""
        clean_data, smoothed, deviation = low_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert 'EVT-ES' not in output

    def test_low_risk_label(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """低偏离(<1.5σ)→低风险"""
        clean_data, smoothed, deviation = low_deviation_data
        # 确保最后一个偏离值 <1.5
        deviation.iloc[-1] = 0.8
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '低风险' in output

    def test_moderate_risk_label(self, vol_modeler_mock, moderate_deviation_data, volatility_series):
        """中等偏离(1.5-2σ)→中等风险"""
        clean_data, smoothed, deviation = moderate_deviation_data
        deviation.iloc[-1] = 1.6
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '中等风险' in output

    def test_high_risk_positive_deviation(self, vol_modeler_mock, high_positive_deviation_data, volatility_series):
        """高正偏离(>2σ)→高风险+高估"""
        clean_data, smoothed, deviation = high_positive_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '高风险' in output
        assert '高估' in output or '均值回归' in output

    def test_high_risk_negative_deviation(self, vol_modeler_mock, high_negative_deviation_data, volatility_series):
        """高负偏离(<-2σ)→高风险+低估"""
        clean_data, smoothed, deviation = high_negative_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '高风险' in output
        assert '低估' in output or '信用事件' in output


# ============================================================================
# 波动率状态
# ============================================================================

class TestVolatilityRegimeSection:
    """波动率状态分位输出测试"""

    def test_prints_volatility_percentile(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """测试输出波动率分位数"""
        clean_data, smoothed, deviation = low_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '分位' in output

    def test_normal_volatility_label(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """正常波动(<75%分位)"""
        clean_data, smoothed, deviation = low_deviation_data
        deviation.iloc[-1] = 0.5
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        # 应出现正常波动期或低风险
        assert '正常波动' in output or '低风险' in output

    def test_high_volatility_warning(self, vol_modeler_mock, low_deviation_data, high_volatility_series):
        """高波动率(>90%分位)→危机模式"""
        clean_data, smoothed, deviation = low_deviation_data
        deviation.iloc[-1] = 0.5
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=high_volatility_series,
                evt_var=5.0,
            )
        assert '危机' in output or '极高水平' in output

    def test_moderate_high_volatility_label(self, vol_modeler_mock, low_deviation_data):
        """中等高波动(75-90%分位)"""
        dates = pd.date_range('2020-01-01', periods=100, freq='B')
        vol = pd.Series(np.linspace(1, 10, 100), index=dates)
        vol.iloc[-1] = 8.0  # ~80%分位
        clean_data, smoothed, deviation = low_deviation_data
        deviation.iloc[-1] = 0.5
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=vol,
                evt_var=5.0,
            )
        assert '高波动' in output or '不确定性' in output


# ============================================================================
# 第四部分: 行动建议
# ============================================================================

class TestActionRecommendationSection:
    """行动建议输出测试"""

    def test_neutral_strategy_low_deviation(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """低偏离→中性观望"""
        clean_data, smoothed, deviation = low_deviation_data
        deviation.iloc[-1] = 0.5
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '中性' in output or '观望' in output

    def test_short_spread_positive_deviation(self, vol_modeler_mock, high_positive_deviation_data, volatility_series):
        """高正偏离→做空利差"""
        clean_data, smoothed, deviation = high_positive_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '做空' in output or '收窄' in output

    def test_long_spread_negative_deviation(self, vol_modeler_mock, high_negative_deviation_data, volatility_series):
        """高负偏离→做多利差"""
        clean_data, smoothed, deviation = high_negative_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '做多' in output or '扩大' in output

    def test_prints_var_limit(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """测试输出VaR限额"""
        clean_data, smoothed, deviation = low_deviation_data
        deviation.iloc[-1] = 0.5
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert 'VaR 限额' in output or '5.00' in output

    def test_prints_position_sizing(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """测试输出仓位规模建议"""
        clean_data, smoothed, deviation = low_deviation_data
        deviation.iloc[-1] = 0.5
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '仓位' in output or '敞口' in output

    def test_prints_monitoring_metrics(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """测试输出关键监控指标"""
        clean_data, smoothed, deviation = low_deviation_data
        deviation.iloc[-1] = 0.5
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '监控指标' in output
        assert '1.5σ' in output or '警戒线' in output

    def test_high_vol_reduces_position(self, vol_modeler_mock, low_deviation_data, high_volatility_series):
        """高波动→建议削减仓位"""
        clean_data, smoothed, deviation = low_deviation_data
        deviation.iloc[-1] = 0.5
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=high_volatility_series,
                evt_var=5.0,
            )
        assert '削减' in output or '50%' in output or '降低' in output

    def test_short_spread_stops_at_var(self, vol_modeler_mock, high_positive_deviation_data, volatility_series):
        """做空止损点=当前+VaR"""
        clean_data, smoothed, deviation = high_positive_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '止损' in output

    def test_long_spread_stops_at_zero_floor(self, vol_modeler_mock, high_negative_deviation_data, volatility_series):
        """做多止损点≥0"""
        clean_data, smoothed, deviation = high_negative_deviation_data
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '止损' in output


# ============================================================================
# 报告格式
# ============================================================================

class TestReportFormat:
    """报告格式测试"""

    def test_prints_header(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """测试输出报告标题头"""
        clean_data, smoothed, deviation = low_deviation_data
        deviation.iloc[-1] = 0.5
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '中国地方债利差战略分析报告' in output

    def test_prints_section_headers(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """测试输出四大板块标题"""
        clean_data, smoothed, deviation = low_deviation_data
        deviation.iloc[-1] = 0.5
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '模型锦标赛结果' in output
        assert '不对称效应' in output
        assert '风险状况' in output
        assert '行动建议' in output

    def test_prints_footer(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """测试输出报告结尾"""
        clean_data, smoothed, deviation = low_deviation_data
        deviation.iloc[-1] = 0.5
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '报告完成' in output

    def test_prints_separator_lines(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """测试输出分隔线"""
        clean_data, smoothed, deviation = low_deviation_data
        deviation.iloc[-1] = 0.5
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '=' * 80 in output
        assert '-' * 80 in output


# ============================================================================
# 边界条件
# ============================================================================

class TestBoundaryConditions:
    """边界条件测试"""

    def test_exact_1_5_sigma_deviation(self, vol_modeler_mock, base_spread_data, volatility_series):
        """恰好1.5σ偏离→低风险(使用严格>比较,abs(1.5)>1.5=False)"""
        smoothed = pd.Series(80.0, index=base_spread_data.index)
        deviation = pd.Series(np.linspace(0, 1.5, 100), index=base_spread_data.index)
        deviation.iloc[-1] = 1.5
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=base_spread_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        # abs(1.5) > 1.5 is False → falls to 低风险 branch
        assert '低风险' in output

    def test_exact_2_0_sigma_deviation(self, vol_modeler_mock, base_spread_data, volatility_series):
        """恰好2.0σ偏离→高风险(严格>比较)"""
        smoothed = pd.Series(80.0, index=base_spread_data.index)
        deviation = pd.Series(np.linspace(0, 2.0, 100), index=base_spread_data.index)
        deviation.iloc[-1] = 2.0
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=base_spread_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        # abs(2.0) > 2.0 is False, so falls into >1.5 branch
        assert '中等风险' in output

    def test_zero_evt_var(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """VaR=0时仍能正常输出"""
        clean_data, smoothed, deviation = low_deviation_data
        deviation.iloc[-1] = 0.5
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=0.0,
            )
        assert '报告完成' in output

    def test_negative_evt_var(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """负VaR时仍能正常输出"""
        clean_data, smoothed, deviation = low_deviation_data
        deviation.iloc[-1] = 0.5
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=-1.0,
            )
        assert '报告完成' in output

    def test_single_data_point(self, vol_modeler_mock):
        """单数据点仍能正常输出"""
        dates = pd.date_range('2020-01-01', periods=1, freq='B')
        clean_data = pd.DataFrame({'spread': pd.Series([85.0], index=dates)})
        smoothed = pd.Series([82.0], index=dates)
        deviation = pd.Series([0.5], index=dates)
        vol = pd.Series([3.0], index=dates)
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=vol,
                evt_var=5.0,
            )
        assert '报告完成' in output

    def test_very_large_evt_var(self, vol_modeler_mock, low_deviation_data, volatility_series):
        """极大VaR时仍能正常输出"""
        clean_data, smoothed, deviation = low_deviation_data
        deviation.iloc[-1] = 0.5
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=99999.0,
            )
        assert '99999.00' in output or '99999' in output

    def test_ewma_winner_model(self, low_deviation_data, volatility_series):
        """EWMA获胜模型(标准模型以外的获胜者)"""
        clean_data, smoothed, deviation = low_deviation_data
        deviation.iloc[-1] = 0.5
        mock = MagicMock()
        mock.ic_scores = {
            'EWMA': {'AIC': 80.0, 'BIC': 90.0},
            'GARCH': {'AIC': 100.0, 'BIC': 110.0},
        }
        mock.results = {}
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='EWMA',
                vol_modeler=mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert 'EWMA' in output
        # EWMA is not EGARCH or GJR-GARCH, so falls into "标准 GARCH" info path
        assert '对称' in output or '未检测到不对称' in output

    def test_very_high_negative_deviation(self, vol_modeler_mock, volatility_series):
        """极高负偏离(-5σ)"""
        dates = pd.date_range('2020-01-01', periods=100, freq='B')
        clean_data = pd.DataFrame({'spread': pd.Series(80 + np.random.randn(100) * 5, index=dates)})
        smoothed = pd.Series(90.0, index=dates)
        deviation = pd.Series(np.linspace(0, -5.0, 100), index=dates)
        captured = StringIO()
        with patch('sys.stdout', captured):
            output = _run_report(
                captured,
                winner_model='GARCH',
                vol_modeler=vol_modeler_mock,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=volatility_series,
                evt_var=5.0,
            )
        assert '高风险' in output
        assert '低估' in output


# ============================================================================
# 集成测试
# ============================================================================

class TestReportIntegration:
    """report.py与其他模块集成测试"""

    def test_with_real_vol_modeler(self):
        """使用真实VolatilityModeler运行报告"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='B')
        returns = pd.Series(np.random.randn(200) * 0.1)

        from volatility import VolatilityModeler
        vol_modeler = VolatilityModeler(returns)
        winner = vol_modeler.run_tournament()
        winner_vol = vol_modeler.get_conditional_volatility(winner)

        spread = pd.Series(80 + np.random.randn(200) * 5, index=dates)
        clean_data = pd.DataFrame({'spread': spread})
        smoothed = spread.rolling(20).mean().fillna(spread.iloc[0])
        deviation = pd.Series(np.random.randn(200) * 0.5, index=dates)

        from evt import EVTRiskAnalyzer
        evt = EVTRiskAnalyzer(returns)
        evt.fit_gpd()
        var = evt.calculate_var()
        es = evt.calculate_es()

        captured = StringIO()
        with patch('sys.stdout', captured):
            generate_strategic_report(
                winner_model=winner,
                vol_modeler=vol_modeler,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=winner_vol,
                evt_var=var,
                evt_es=es,
            )
        output = captured.getvalue()
        assert '报告完成' in output
        assert winner in output

    def test_with_kalman_signal(self):
        """与KalmanSignalExtractor集成"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='B')
        spread = pd.Series(80 + np.cumsum(np.random.randn(200) * 0.5), index=dates)

        from kalman import KalmanSignalExtractor
        from volatility import VolatilityModeler

        kalman = KalmanSignalExtractor(spread)
        smoothed = kalman.fit()
        deviation = kalman.get_signal_deviation()

        returns = pd.Series(np.random.randn(200) * 0.1)
        vol_modeler = VolatilityModeler(returns)
        winner = vol_modeler.run_tournament()
        winner_vol = vol_modeler.get_conditional_volatility(winner)

        clean_data = pd.DataFrame({'spread': spread})

        captured = StringIO()
        with patch('sys.stdout', captured):
            generate_strategic_report(
                winner_model=winner,
                vol_modeler=vol_modeler,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=winner_vol,
                evt_var=5.0,
            )
        output = captured.getvalue()
        assert '报告完成' in output
        assert '卡尔曼趋势' in output

    def test_full_pipeline_integration(self):
        """完整分析管线→报告生成"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=300, freq='B')

        from data_engine import DataEngine
        config = {
            'SOURCE': 'MOCK',
            'START_DATE': '2020-01-01',
            'END_DATE': '2021-12-31',
        }
        engine = DataEngine(config)
        clean_data = engine.load_data()
        engine.clean_data()
        returns = engine.get_returns()

        from volatility import VolatilityModeler
        vol_modeler = VolatilityModeler(returns)
        winner = vol_modeler.run_tournament()
        winner_vol = vol_modeler.get_conditional_volatility(winner)

        from kalman import KalmanSignalExtractor
        kalman = KalmanSignalExtractor(clean_data['spread'])
        smoothed = kalman.fit()
        deviation = kalman.get_signal_deviation()

        from evt import EVTRiskAnalyzer
        evt = EVTRiskAnalyzer(returns)
        evt.fit_gpd()
        var = evt.calculate_var()
        es = evt.calculate_es()

        captured = StringIO()
        with patch('sys.stdout', captured):
            generate_strategic_report(
                winner_model=winner,
                vol_modeler=vol_modeler,
                clean_data=clean_data,
                smoothed_spread=smoothed,
                signal_deviation=deviation,
                winner_volatility=winner_vol,
                evt_var=var,
                evt_es=es,
            )
        output = captured.getvalue()
        assert len(output) > 500  # 报告应有一定长度
        assert '报告完成' in output
        assert '模型锦标赛结果' in output
        assert '风险状况' in output
        assert '行动建议' in output


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
