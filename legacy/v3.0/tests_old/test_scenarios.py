"""
情景分析模块单元测试 - 测试 scenarios.py 的所有核心功能
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scenarios import (
    run_stress_test,
    run_multi_scenario_stress,
    run_monte_carlo,
    plot_mc_simulation,
    plot_mc_paths,
    run_sensitivity_analysis,
    plot_sensitivity_analysis,
    calculate_rolling_stats,
    detect_historical_events,
    plot_rolling_stats,
    plot_percentile_chart
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_returns():
    """生成测试用收益率数据"""
    np.random.seed(42)
    return pd.Series(np.random.standard_t(5, 500) * 0.5)


@pytest.fixture
def sample_data():
    """生成测试用利差数据"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=300, freq='B')
    spread = 80 + np.cumsum(np.random.randn(300) * 0.3)
    return pd.DataFrame({'spread': spread}, index=dates)


@pytest.fixture
def mc_results(sample_returns):
    """生成蒙特卡洛模拟结果"""
    return run_monte_carlo(sample_returns, n_simulations=1000, horizon=10, seed=42)


# ============================================================================
# run_stress_test 测试
# ============================================================================

class TestStressTest:

    def test_returns_dict(self, sample_returns):
        """测试返回字典"""
        result = run_stress_test(sample_returns, shock=5.0)
        assert isinstance(result, dict)

    def test_has_required_fields(self, sample_returns):
        """测试包含必要字段"""
        result = run_stress_test(sample_returns, shock=5.0)
        assert 'var' in result
        assert 'es' in result
        assert 'max_loss' in result
        assert 'original_var' in result
        assert 'var_change' in result
        assert 'shock' in result

    def test_var_positive(self, sample_returns):
        """测试 VaR 为正数"""
        result = run_stress_test(sample_returns, shock=5.0)
        assert result['var'] > 0

    def test_es_greater_than_var(self, sample_returns):
        """测试 ES >= VaR"""
        result = run_stress_test(sample_returns, shock=5.0)
        assert result['es'] >= result['var']

    def test_shock_value_preserved(self, sample_returns):
        """测试冲击值保留"""
        result = run_stress_test(sample_returns, shock=5.0)
        assert result['shock'] == 5.0

    def test_var_change_positive(self, sample_returns):
        """测试 VaR 变化为正"""
        result = run_stress_test(sample_returns, shock=5.0)
        assert result['var_change'] > 0

    def test_stressed_var_greater_than_original(self, sample_returns):
        """测试压力 VaR 大于原始 VaR"""
        result = run_stress_test(sample_returns, shock=5.0)
        assert result['var'] > result['original_var']

    def test_negative_shock(self, sample_returns):
        """测试负冲击"""
        result = run_stress_test(sample_returns, shock=-5.0)
        assert isinstance(result, dict)
        assert result['shock'] == -5.0

    def test_zero_shock(self, sample_returns):
        """测试零冲击"""
        result = run_stress_test(sample_returns, shock=0.0)
        assert result['var'] > 0


# ============================================================================
# run_multi_scenario_stress 测试
# ============================================================================

class TestMultiScenarioStress:

    def test_returns_dataframe(self, sample_returns):
        """测试返回 DataFrame"""
        result = run_multi_scenario_stress(sample_returns)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, sample_returns):
        """测试包含预期列"""
        result = run_multi_scenario_stress(sample_returns)
        assert 'shock' in result.columns
        assert 'var' in result.columns
        assert 'es' in result.columns
        assert 'max_loss' in result.columns

    def test_default_scenarios_count(self, sample_returns):
        """测试默认情景数"""
        result = run_multi_scenario_stress(sample_returns)
        assert len(result) == 21

    def test_custom_scenario_count(self, sample_returns):
        """测试自定义情景数"""
        result = run_multi_scenario_stress(sample_returns, n_scenarios=11)
        assert len(result) == 11

    def test_custom_shock_range(self, sample_returns):
        """测试自定义冲击范围"""
        result = run_multi_scenario_stress(sample_returns, shock_range=(-20, 20))
        assert result['shock'].min() == -20
        assert result['shock'].max() == 20

    def test_var_monotonic_with_shock(self, sample_returns):
        """测试 VaR 随冲击单调递增"""
        result = run_multi_scenario_stress(sample_returns, n_scenarios=11)
        # 对于正冲击，VaR 应随冲击增大
        positive = result[result['shock'] >= 0].sort_values('shock')
        vars = positive['var'].values
        for i in range(len(vars) - 1):
            assert vars[i + 1] >= vars[i]


# ============================================================================
# run_monte_carlo 测试
# ============================================================================

class TestMonteCarlo:

    def test_returns_dict(self, sample_returns):
        """测试返回字典"""
        result = run_monte_carlo(sample_returns, seed=42)
        assert isinstance(result, dict)

    def test_has_required_fields(self, mc_results):
        """测试包含必要字段"""
        assert 'paths' in mc_results
        assert 'final_values' in mc_results
        assert 'mean' in mc_results
        assert 'std' in mc_results
        assert 'var_95' in mc_results
        assert 'var_99' in mc_results
        assert 'es_99' in mc_results
        assert 'min' in mc_results
        assert 'max' in mc_results
        assert 'median' in mc_results
        assert 'params' in mc_results

    def test_paths_shape(self, mc_results):
        """测试模拟路径形状"""
        # paths shape: (n_simulations, horizon+1)
        assert mc_results['paths'].shape[0] == 1000
        assert mc_results['paths'].shape[1] == 11  # horizon=10, +1 for initial

    def test_params_include_ar(self, mc_results):
        """测试参数包含 AR(1) 系数"""
        assert 'phi' in mc_results['params']
        assert 'mu' in mc_results['params']
        assert 'sigma' in mc_results['params']
        assert 'df' in mc_results['params']

    def test_phi_bounded(self, mc_results):
        """测试 phi 在合理范围"""
        phi = mc_results['params']['phi']
        assert 0 <= phi <= 0.99

    def test_var_99_greater_than_var_95(self, mc_results):
        """测试 VaR99 >= VaR95"""
        assert mc_results['var_99'] >= mc_results['var_95']

    def test_es_greater_than_var_99(self, mc_results):
        """测试 ES99 >= VaR99"""
        assert mc_results['es_99'] >= mc_results['var_99']

    def test_reproducible_with_seed(self, sample_returns):
        """测试种子可复现"""
        r1 = run_monte_carlo(sample_returns, seed=42)
        r2 = run_monte_carlo(sample_returns, seed=42)
        assert r1['mean'] == r2['mean']
        assert r1['var_99'] == r2['var_99']

    def test_custom_horizon(self, sample_returns):
        """测试自定义预测天数"""
        result = run_monte_carlo(sample_returns, horizon=20, seed=42)
        assert result['paths'].shape[1] == 21

    def test_custom_simulations(self, sample_returns):
        """测试自定义模拟次数"""
        result = run_monte_carlo(sample_returns, n_simulations=500, seed=42)
        assert result['paths'].shape[0] == 500

    def test_initial_value(self, sample_returns):
        """测试初始值设置"""
        result = run_monte_carlo(sample_returns, seed=42, initial_value=5.0)
        assert result['paths'][0, 0] == 5.0


# ============================================================================
# run_sensitivity_analysis 测试
# ============================================================================

class TestSensitivityAnalysis:

    def test_volatility_returns_dataframe(self, sample_returns):
        """测试波动率敏感性返回 DataFrame"""
        result = run_sensitivity_analysis(sample_returns, param='volatility')
        assert isinstance(result, pd.DataFrame)

    def test_volatility_columns(self, sample_returns):
        """测试波动率敏感性列"""
        result = run_sensitivity_analysis(sample_returns, param='volatility')
        assert 'parameter' in result.columns
        assert 'value' in result.columns
        assert 'var_99' in result.columns
        assert 'es_99' in result.columns

    def test_mean_sensitivity(self, sample_returns):
        """测试均值敏感性"""
        result = run_sensitivity_analysis(sample_returns, param='mean')
        assert len(result) == 20
        assert all(result['parameter'] == 'mean')

    def test_df_sensitivity(self, sample_returns):
        """测试自由度敏感性"""
        result = run_sensitivity_analysis(sample_returns, param='df')
        assert len(result) == 20
        assert all(result['parameter'] == 'df')

    def test_var_increases_with_volatility(self, sample_returns):
        """测试 VaR 随波动率增大（允许模拟误差）"""
        result = run_sensitivity_analysis(sample_returns, param='volatility', n_points=10)
        sorted_result = result.sort_values('value')
        vars = sorted_result['var_99'].values
        first_var = vars[0]
        last_var = vars[-1]
        assert last_var > first_var  # 总体趋势：最高波动率 VaR > 最低波动率 VaR

    def test_custom_n_points(self, sample_returns):
        """测试自定义分析点数"""
        result = run_sensitivity_analysis(sample_returns, param='volatility', n_points=10)
        assert len(result) == 10


# ============================================================================
# calculate_rolling_stats 测试
# ============================================================================

class TestRollingStats:

    def test_returns_dataframe(self, sample_data):
        """测试返回 DataFrame"""
        result = calculate_rolling_stats(sample_data)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, sample_data):
        """测试包含预期列"""
        result = calculate_rolling_stats(sample_data)
        expected_cols = ['rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max',
                            'rolling_range', 'rolling_q25', 'rolling_q75', 'rolling_iqr',
                            'rolling_skew', 'rolling_kurt']
        for col in expected_cols:
            assert col in result.columns

    def test_index_preserved(self, sample_data):
        """测试索引保留"""
        result = calculate_rolling_stats(sample_data)
        assert result.index.equals(sample_data.index)

    def test_custom_window(self, sample_data):
        """测试自定义窗口"""
        result = calculate_rolling_stats(sample_data, window=30)
        assert isinstance(result, pd.DataFrame)
        # 前29行应该是 NaN
        assert result['rolling_mean'].iloc[:29].isna().all()

    def test_range_equals_max_minus_min(self, sample_data):
        """测试范围 = 最大 - 最小"""
        result = calculate_rolling_stats(sample_data, window=60)
        valid = result.dropna()
        assert np.allclose(
            valid['rolling_range'],
            valid['rolling_max'] - valid['rolling_min'],
            atol=1e-10
        )


# ============================================================================
# detect_historical_events 测试
# ============================================================================

class TestEventDetection:

    def test_returns_dataframe(self, sample_data):
        """测试返回 DataFrame"""
        result = detect_historical_events(sample_data)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, sample_data):
        """测试包含预期列"""
        result = detect_historical_events(sample_data)
        expected_cols = ['日期', '类型', '描述', '影响']
        for col in expected_cols:
            assert col in result.columns

    def test_max_20_events(self, sample_data):
        """测试最多20个事件"""
        result = detect_historical_events(sample_data)
        assert len(result) <= 20

    def test_custom_threshold(self, sample_data):
        """测试自定义阈值"""
        result_low = detect_historical_events(sample_data, threshold=2.0)
        result_high = detect_historical_events(sample_data, threshold=5.0)
        assert len(result_low) >= len(result_high)

    def test_event_types(self, sample_data):
        """测试事件类型"""
        result = detect_historical_events(sample_data, threshold=2.0)
        valid_types = {'异常值', '大幅波动'}
        for event_type in result['类型']:
            assert event_type in valid_types


# ============================================================================
# 蒙特卡洛可视化测试
# ============================================================================

class TestMCVisualization:

    def test_plot_mc_simulation_returns_figure(self, mc_results):
        """测试 MC 分布图返回 Figure"""
        fig = plot_mc_simulation(mc_results)
        assert isinstance(fig, go.Figure)

    def test_plot_mc_simulation_dark_theme(self, mc_results):
        """测试 MC 分布图深色主题"""
        fig = plot_mc_simulation(mc_results, theme='dark')
        assert isinstance(fig, go.Figure)

    def test_plot_mc_paths_returns_figure(self, mc_results):
        """测试 MC 路径图返回 Figure"""
        fig = plot_mc_paths(mc_results)
        assert isinstance(fig, go.Figure)

    def test_plot_mc_paths_custom_n_paths(self, mc_results):
        """测试 MC 路径图自定义路径数"""
        fig = plot_mc_paths(mc_results, n_paths=10)
        assert isinstance(fig, go.Figure)

    def test_plot_mc_paths_dark_theme(self, mc_results):
        """测试 MC 路径图深色主题"""
        fig = plot_mc_paths(mc_results, theme='dark')
        assert isinstance(fig, go.Figure)


# ============================================================================
# 敏感性可视化测试
# ============================================================================

class TestSensitivityVisualization:

    def test_plot_sensitivity_returns_figure(self, sample_returns):
        """测试敏感性图返回 Figure"""
        result = run_sensitivity_analysis(sample_returns, param='volatility')
        fig = plot_sensitivity_analysis(result, param='volatility')
        assert isinstance(fig, go.Figure)

    def test_plot_sensitivity_mean(self, sample_returns):
        """测试均值敏感性图"""
        result = run_sensitivity_analysis(sample_returns, param='mean')
        fig = plot_sensitivity_analysis(result, param='mean')
        assert isinstance(fig, go.Figure)

    def test_plot_sensitivity_dark_theme(self, sample_returns):
        """测试敏感性图深色主题"""
        result = run_sensitivity_analysis(sample_returns, param='volatility')
        fig = plot_sensitivity_analysis(result, param='volatility', theme='dark')
        assert isinstance(fig, go.Figure)


# ============================================================================
# 滚动统计可视化测试
# ============================================================================

class TestRollingVisualization:

    def test_plot_rolling_stats_returns_figure(self, sample_data):
        """测试滚动统计图返回 Figure"""
        result = calculate_rolling_stats(sample_data)
        fig = plot_rolling_stats(result)
        assert isinstance(fig, go.Figure)

    def test_plot_rolling_stats_with_original(self, sample_data):
        """测试滚动统计图带原始数据"""
        result = calculate_rolling_stats(sample_data)
        fig = plot_rolling_stats(result, original_data=sample_data)
        assert isinstance(fig, go.Figure)

    def test_plot_rolling_stats_dark_theme(self, sample_data):
        """测试滚动统计图深色主题"""
        result = calculate_rolling_stats(sample_data)
        fig = plot_rolling_stats(result, theme='dark')
        assert isinstance(fig, go.Figure)

    def test_plot_percentile_returns_figure(self, sample_data):
        """测试分位数图返回 Figure"""
        fig = plot_percentile_chart(sample_data)
        assert isinstance(fig, go.Figure)

    def test_plot_percentile_custom_windows(self, sample_data):
        """测试分位数图自定义窗口"""
        fig = plot_percentile_chart(sample_data, windows=[20, 60])
        assert isinstance(fig, go.Figure)

    def test_plot_percentile_dark_theme(self, sample_data):
        """测试分位数图深色主题"""
        fig = plot_percentile_chart(sample_data, theme='dark')
        assert isinstance(fig, go.Figure)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
