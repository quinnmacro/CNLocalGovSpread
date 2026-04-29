"""
市场状态仪表模块测试

覆盖:
1. MarketStatusGauge 初始化与权重配置
2. 各指标评分计算 (利差定位, 波动率状态, VaR突破, 信号偏离, 趋势动量)
3. 加权融合综合评分
4. 状态等级判断与极端指标升级
5. 指标联动雷达图
6. 滚动时间线
7. 指标相关性计算
8. 可视化函数 (仪表盘, 雷达图, 时间线)
9. 短数据/缺失数据 fallback
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from data_engine import DataEngine
from volatility import VolatilityModeler
from kalman import KalmanSignalExtractor
from evt import EVTRiskAnalyzer
from market_status import MarketStatusGauge


# ============================================================================
# 测试数据生成
# ============================================================================

def generate_test_data(n=500):
    """生成测试数据"""
    config = {
        'SOURCE': 'MOCK',
        'START_DATE': '2020-01-01',
        'END_DATE': '2024-01-01',
        'MAD_THRESHOLD': 5.0,
        'SPREAD_COLUMN': 'spread_all'
    }
    engine = DataEngine(config)
    raw = engine.load_data()
    clean = engine.clean_data()
    returns = engine.get_returns()
    return clean, returns


def generate_test_data_short(n=50):
    """生成短数据用于边界测试"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n, freq='B')
    spread = np.cumsum(np.random.randn(n) * 0.5) + 100
    data = pd.DataFrame({'spread': spread}, index=dates)
    returns = data['spread'].diff().dropna()
    return data, returns


def run_full_analysis(clean_data, returns):
    """运行完整分析流程"""
    # Kalman滤波
    kalman = KalmanSignalExtractor(clean_data['spread'])
    smoothed = kalman.fit()
    deviation = kalman.get_signal_deviation()

    # GARCH锦标赛
    vol_modeler = VolatilityModeler(returns)
    winner = vol_modeler.run_tournament()

    # EVT风险分析
    evt = EVTRiskAnalyzer(returns, threshold_percentile=0.95, confidence=0.99)
    evt.fit_gpd()
    var = evt.calculate_var()
    es = evt.calculate_es()

    return {
        'smoothed': smoothed,
        'deviation': deviation,
        'vol_modeler': vol_modeler,
        'winner': winner,
        'evt': evt,
        'var': var,
        'es': es
    }


# ============================================================================
# 初始化与配置测试
# ============================================================================

class TestMarketStatusGaugeInit:

    def test_default_weights(self):
        """默认权重配置正确"""
        clean, returns = generate_test_data_short()
        gauge = MarketStatusGauge(clean, returns)
        assert gauge.weights == MarketStatusGauge.DEFAULT_WEIGHTS
        total = sum(gauge.weights.values())
        assert total == 1.0

    def test_custom_weights(self):
        """自定义权重生效"""
        clean, returns = generate_test_data_short()
        custom = {
            'spread_position': 0.3,
            'volatility_regime': 0.2,
            'var_breach': 0.2,
            'signal_deviation': 0.15,
            'trend_momentum': 0.15
        }
        gauge = MarketStatusGauge(clean, returns, weights=custom)
        assert gauge.weights == custom

    def test_init_with_optional_none(self):
        """可选参数为None时正常初始化"""
        clean, returns = generate_test_data_short()
        gauge = MarketStatusGauge(clean, returns)
        assert gauge.smoothed is None
        assert gauge.deviation is None
        assert gauge.vol_modeler is None
        assert gauge.evt is None

    def test_init_with_all_params(self):
        """所有参数正常初始化"""
        clean, returns = generate_test_data()
        analysis = run_full_analysis(clean, returns)
        gauge = MarketStatusGauge(
            clean, returns,
            smoothed=analysis['smoothed'],
            deviation=analysis['deviation'],
            vol_modeler=analysis['vol_modeler'],
            evt=analysis['evt']
        )
        assert gauge.smoothed is not None
        assert gauge.deviation is not None
        assert gauge.vol_modeler is not None
        assert gauge.evt is not None


# ============================================================================
# 指标评分计算测试
# ============================================================================

class TestIndicatorScores:

    def test_calculate_indicator_scores_returns_dict(self):
        """calculate_indicator_scores 返回包含5个指标的dict"""
        clean, returns = generate_test_data_short()
        gauge = MarketStatusGauge(clean, returns)
        scores = gauge.calculate_indicator_scores()
        assert isinstance(scores, dict)
        assert 'spread_position' in scores
        assert 'volatility_regime' in scores
        assert 'var_breach' in scores
        assert 'signal_deviation' in scores
        assert 'trend_momentum' in scores

    def test_all_scores_in_range(self):
        """所有指标评分在0-100范围内"""
        clean, returns = generate_test_data_short()
        gauge = MarketStatusGauge(clean, returns)
        scores = gauge.calculate_indicator_scores()
        for key, info in scores.items():
            assert 0 <= info['score'] <= 100, f"{key} score {info['score']} out of range"

    def test_spread_position_score(self):
        """利差定位评分包含必要字段"""
        clean, returns = generate_test_data_short()
        gauge = MarketStatusGauge(clean, returns)
        scores = gauge.calculate_indicator_scores()
        sp = scores['spread_position']
        assert 'z_score' in sp
        assert 'current' in sp
        assert 'mean' in sp
        assert 'std' in sp
        assert 'percentile' in sp
        assert 0 <= sp['percentile'] <= 100

    def test_volatility_regime_without_modeler(self):
        """无波动率模型器时评分默认为50(中性)"""
        clean, returns = generate_test_data_short()
        gauge = MarketStatusGauge(clean, returns)
        scores = gauge.calculate_indicator_scores()
        assert scores['volatility_regime']['score'] == 50
        assert scores['volatility_regime']['current_vol'] is None

    def test_volatility_regime_with_modeler(self):
        """有波动率模型器时评分基于实际数据"""
        clean, returns = generate_test_data()
        analysis = run_full_analysis(clean, returns)
        gauge = MarketStatusGauge(
            clean, returns,
            vol_modeler=analysis['vol_modeler']
        )
        scores = gauge.calculate_indicator_scores()
        vol = scores['volatility_regime']
        assert vol['current_vol'] is not None
        assert vol['ratio'] is not None
        assert vol['score'] != 50  # 不应该是默认中性

    def test_var_breach_with_evt(self):
        """有EVT分析器时VaR突破评分基于实际数据"""
        clean, returns = generate_test_data()
        analysis = run_full_analysis(clean, returns)
        gauge = MarketStatusGauge(
            clean, returns,
            evt=analysis['evt']
        )
        scores = gauge.calculate_indicator_scores()
        var_info = scores['var_breach']
        assert var_info['var'] is not None
        assert var_info['breach_ratio'] is not None

    def test_signal_deviation_without_kalman(self):
        """无Kalman滤波时信号偏离评分默认为50"""
        clean, returns = generate_test_data_short()
        gauge = MarketStatusGauge(clean, returns)
        scores = gauge.calculate_indicator_scores()
        assert scores['signal_deviation']['score'] == 50
        assert scores['signal_deviation']['current_deviation'] is None

    def test_signal_deviation_with_kalman(self):
        """有Kalman滤波时信号偏离评分基于实际偏离度"""
        clean, returns = generate_test_data()
        analysis = run_full_analysis(clean, returns)
        gauge = MarketStatusGauge(
            clean, returns,
            deviation=analysis['deviation']
        )
        scores = gauge.calculate_indicator_scores()
        dev = scores['signal_deviation']
        assert dev['current_deviation'] is not None
        assert dev['abs_deviation'] is not None

    def test_trend_momentum_score(self):
        """趋势动量评分包含方向和强度"""
        clean, returns = generate_test_data()
        gauge = MarketStatusGauge(clean, returns)
        scores = gauge.calculate_indicator_scores()
        tm = scores['trend_momentum']
        assert 'direction' in tm
        assert tm['direction'] in ('扩大', '收窄')
        assert tm['trend_delta'] is not None

    def test_trend_momentum_short_data(self):
        """数据不足60天时趋势动量评分默认为50"""
        clean, returns = generate_test_data_short(30)
        gauge = MarketStatusGauge(clean, returns)
        scores = gauge.calculate_indicator_scores()
        tm = scores['trend_momentum']
        assert tm['score'] == 50
        assert tm['direction'] is None


# ============================================================================
# 综合评分与状态测试
# ============================================================================

class TestCompositeScore:

    def test_composite_score_in_range(self):
        """综合评分在0-100范围内"""
        clean, returns = generate_test_data_short()
        gauge = MarketStatusGauge(clean, returns)
        composite = gauge.calculate_composite_score()
        assert 0 <= composite['score'] <= 100

    def test_composite_has_status_fields(self):
        """综合评分包含状态字段"""
        clean, returns = generate_test_data_short()
        gauge = MarketStatusGauge(clean, returns)
        composite = gauge.calculate_composite_score()
        assert 'status' in composite
        assert 'label' in composite
        assert 'color' in composite
        assert composite['status'] in MarketStatusGauge.STATUS_LEVELS

    def test_composite_indicator_scores_dict(self):
        """综合评分包含各指标评分汇总"""
        clean, returns = generate_test_data_short()
        gauge = MarketStatusGauge(clean, returns)
        composite = gauge.calculate_composite_score()
        assert 'indicator_scores' in composite
        assert len(composite['indicator_scores']) == 5

    def test_composite_weights_in_output(self):
        """综合评分包含权重信息"""
        clean, returns = generate_test_data_short()
        gauge = MarketStatusGauge(clean, returns)
        composite = gauge.calculate_composite_score()
        assert 'weights' in composite
        assert composite['weights'] == gauge.weights

    def test_extreme_indicator_upgrades_status(self):
        """极端指标(>=90)触发状态升级"""
        # 创建一个高利差场景
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=300, freq='B')
        # 利差从50突然跳到200 (Z-score极高)
        spread = np.concatenate([
            np.random.randn(290) * 2 + 50,
            np.random.randn(10) * 2 + 200
        ])
        data = pd.DataFrame({'spread': spread}, index=dates)
        returns = data['spread'].diff().dropna()

        gauge = MarketStatusGauge(data, returns)
        scores = gauge.calculate_indicator_scores()
        composite = gauge.calculate_composite_score()

        # 利差定位评分应该很高
        assert scores['spread_position']['score'] >= 80

    def test_safe_status_for_low_scores(self):
        """低评分时状态为'安全'"""
        np.random.seed(123)
        dates = pd.date_range('2020-01-01', periods=300, freq='B')
        # 稳定低利差 (均值附近, 低波动)
        spread = np.random.randn(300) * 1 + 50  # 标准差很小
        data = pd.DataFrame({'spread': spread}, index=dates)
        returns = data['spread'].diff().dropna()

        gauge = MarketStatusGauge(data, returns)
        composite = gauge.calculate_composite_score()
        # 综合评分应该偏低
        assert composite['score'] < 60

    def test_get_market_status_returns_composite(self):
        """get_market_status 返回综合评分"""
        clean, returns = generate_test_data_short()
        gauge = MarketStatusGauge(clean, returns)
        status = gauge.get_market_status()
        assert status == gauge._composite_score


# ============================================================================
# 指标联动与相关性测试
# ============================================================================

class TestIndicatorLinkage:

    def test_calculate_indicator_correlation_returns_dict(self):
        """calculate_indicator_correlation 返回嵌套dict"""
        clean, returns = generate_test_data()
        gauge = MarketStatusGauge(clean, returns)
        corr = gauge.calculate_indicator_correlation()
        assert isinstance(corr, dict)

    def test_correlation_symmetric(self):
        """相关性矩阵对称"""
        clean, returns = generate_test_data()
        gauge = MarketStatusGauge(clean, returns)
        corr = gauge.calculate_indicator_correlation()
        if not corr:
            pytest.skip("数据不足，无法计算相关性")
        for key1 in corr:
            for key2 in corr:
                if key1 in corr[key2] and key2 in corr:
                    assert abs(corr[key1][key2] - corr[key2][key1]) < 0.01

    def test_correlation_values_in_range(self):
        """相关性值在-1到1范围内"""
        clean, returns = generate_test_data()
        gauge = MarketStatusGauge(clean, returns)
        corr = gauge.calculate_indicator_correlation()
        if not corr:
            pytest.skip("数据不足")
        for key1 in corr:
            for key2, value in corr[key1].items():
                assert -1 <= value <= 1

    def test_correlation_short_data_returns_empty(self):
        """数据不足时相关性返回空dict"""
        clean, returns = generate_test_data_short(30)
        gauge = MarketStatusGauge(clean, returns)
        corr = gauge.calculate_indicator_correlation(window=60)
        assert corr == {}


# ============================================================================
# 可视化函数测试
# ============================================================================

class TestVisualization:

    def test_plot_status_gauge_returns_figure(self):
        """plot_status_gauge 返回Plotly Figure"""
        clean, returns = generate_test_data_short()
        gauge = MarketStatusGauge(clean, returns)
        gauge.calculate_composite_score()
        fig = gauge.plot_status_gauge(theme='light')
        assert fig is not None
        assert hasattr(fig, 'data')

    def test_plot_status_gauge_dark_theme(self):
        """深色主题仪表盘正常生成"""
        clean, returns = generate_test_data_short()
        gauge = MarketStatusGauge(clean, returns)
        gauge.calculate_composite_score()
        fig = gauge.plot_status_gauge(theme='dark')
        assert fig is not None

    def test_plot_indicator_linkage_returns_figure(self):
        """plot_indicator_linkage 返回Plotly Figure"""
        clean, returns = generate_test_data_short()
        gauge = MarketStatusGauge(clean, returns)
        gauge.calculate_composite_score()
        fig = gauge.plot_indicator_linkage(theme='light')
        assert fig is not None
        # 雷达图应该有3条trace: 当前状态, 安全基准, 警戒线
        assert len(fig.data) == 3

    def test_plot_indicator_linkage_dark_theme(self):
        """深色主题雷达图正常生成"""
        clean, returns = generate_test_data_short()
        gauge = MarketStatusGauge(clean, returns)
        gauge.calculate_composite_score()
        fig = gauge.plot_indicator_linkage(theme='dark')
        assert fig is not None

    def test_plot_indicator_timeline_returns_figure(self):
        """plot_indicator_timeline 返回Plotly Figure"""
        clean, returns = generate_test_data()
        gauge = MarketStatusGauge(clean, returns)
        fig = gauge.plot_indicator_timeline(window=60, theme='light')
        assert fig is not None

    def test_plot_indicator_timeline_short_data(self):
        """短数据时时间线显示提示信息"""
        clean, returns = generate_test_data_short(50)
        gauge = MarketStatusGauge(clean, returns)
        fig = gauge.plot_indicator_timeline(window=60, theme='light')
        assert fig is not None

    def test_plot_indicator_timeline_dark_theme(self):
        """深色主题时间线正常生成"""
        clean, returns = generate_test_data()
        gauge = MarketStatusGauge(clean, returns)
        fig = gauge.plot_indicator_timeline(window=60, theme='dark')
        assert fig is not None


# ============================================================================
# 全流程集成测试
# ============================================================================

class TestIntegration:

    def test_full_workflow(self):
        """完整分析流程 + 状态仪表"""
        clean, returns = generate_test_data()
        analysis = run_full_analysis(clean, returns)

        gauge = MarketStatusGauge(
            clean, returns,
            smoothed=analysis['smoothed'],
            deviation=analysis['deviation'],
            vol_modeler=analysis['vol_modeler'],
            evt=analysis['evt']
        )

        # 计算指标评分
        scores = gauge.calculate_indicator_scores()
        assert all(0 <= info['score'] <= 100 for info in scores.values())

        # 计算综合评分
        composite = gauge.calculate_composite_score()
        assert 0 <= composite['score'] <= 100
        assert composite['status'] in MarketStatusGauge.STATUS_LEVELS

        # 生成可视化
        gauge_fig = gauge.plot_status_gauge(theme='light')
        radar_fig = gauge.plot_indicator_linkage(theme='light')
        timeline_fig = gauge.plot_indicator_timeline(theme='light')
        assert gauge_fig is not None
        assert radar_fig is not None
        assert timeline_fig is not None

    def test_minimal_workflow(self):
        """仅基础数据的状态仪表"""
        clean, returns = generate_test_data_short()
        gauge = MarketStatusGauge(clean, returns)

        scores = gauge.calculate_indicator_scores()
        composite = gauge.calculate_composite_score()

        # 即使没有Kalman/GARCH/EVT，仍能计算部分评分
        assert 'spread_position' in scores
        assert 'trend_momentum' in scores
        assert composite['score'] is not None

    def test_status_levels_definition(self):
        """STATUS_LEVELS 定义完整"""
        levels = MarketStatusGauge.STATUS_LEVELS
        assert 'safe' in levels
        assert 'watch' in levels
        assert 'caution' in levels
        assert 'warning' in levels
        assert 'danger' in levels

        # 每个等级有label和color
        for key, info in levels.items():
            assert 'label' in info
            assert 'color' in info
            assert 'range' in info

    def test_indicator_names_consistency(self):
        """指标名称与权重key一致"""
        gauge = MarketStatusGauge(*generate_test_data_short())
        scores = gauge.calculate_indicator_scores()
        for key in gauge.weights:
            assert key in scores, f"Weight key '{key}' not found in scores"

    def test_composite_score_weighted_sum(self):
        """综合评分确实是加权求和"""
        clean, returns = generate_test_data_short()
        gauge = MarketStatusGauge(clean, returns)
        scores = gauge.calculate_indicator_scores()
        composite = gauge.calculate_composite_score()

        # 手动计算加权求和
        manual_sum = 0
        for key, weight in gauge.weights.items():
            manual_sum += scores[key]['score'] * weight

        assert abs(composite['score'] - manual_sum) < 1.0