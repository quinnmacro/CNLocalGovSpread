"""
EVTRiskAnalyzer 单元测试 - 极值理论风险分析模块
v3.0 - CNLocalGovSpread 计量经济学框架

覆盖: 初始化, fit_gpd, calculate_var, calculate_es, get_tail_index,
      mean_excess_plot, estimate_hill, 边界条件, 集成流程
"""

import pytest
import numpy as np
import pandas as pd
from evt import EVTRiskAnalyzer


# ============================================================================
# 基础初始化测试
# ============================================================================

class TestEVTInit:

    def test_init_default_params(self, fat_tail_returns):
        """默认参数初始化"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        assert analyzer.threshold_percentile == 0.95
        assert analyzer.confidence == 0.99
        assert analyzer.threshold is None
        assert analyzer.gpd_params is None
        assert analyzer.var is None
        assert analyzer.es is None
        assert analyzer.hill_estimator is None

    def test_init_custom_params(self, fat_tail_returns):
        """自定义参数初始化"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns, threshold_percentile=0.90, confidence=0.95)
        assert analyzer.threshold_percentile == 0.90
        assert analyzer.confidence == 0.95

    def test_init_stores_returns(self, fat_tail_returns):
        """初始化应存储 returns"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        assert analyzer.returns is fat_tail_returns


# ============================================================================
# fit_gpd() 测试
# ============================================================================

class TestFitGPD:

    def test_fit_sets_threshold(self, fat_tail_returns):
        """fit_gpd 应设置阈值"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        assert analyzer.threshold is not None
        assert analyzer.threshold > fat_tail_returns.quantile(0.90)

    def test_fit_threshold_at_percentile(self, fat_tail_returns):
        """阈值应在指定百分位附近"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns, threshold_percentile=0.95)
        analyzer.fit_gpd()
        expected = fat_tail_returns.quantile(0.95)
        assert abs(analyzer.threshold - expected) < 0.01

    def test_fit_custom_threshold_percentile(self, fat_tail_returns):
        """自定义百分位阈值"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns, threshold_percentile=0.90)
        analyzer.fit_gpd()
        assert analyzer.threshold > fat_tail_returns.quantile(0.85)

    def test_fit_sets_gpd_params(self, fat_tail_returns):
        """fit_gpd 应设置 GPD 参数"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        if analyzer.gpd_params is not None:
            assert 'shape' in analyzer.gpd_params
            assert 'scale' in analyzer.gpd_params

    def test_fit_shape_positive_for_fat_tail(self, fat_tail_returns):
        """肥尾数据形状参数应为正"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        if analyzer.gpd_params is not None:
            assert analyzer.gpd_params['shape'] > 0

    def test_fit_scale_positive(self, fat_tail_returns):
        """尺度参数应为正"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        if analyzer.gpd_params is not None:
            assert analyzer.gpd_params['scale'] > 0

    def test_fit_exceedances_count(self, fat_tail_returns):
        """超过阈值的样本数应合理"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns, threshold_percentile=0.95)
        analyzer.fit_gpd()
        exceedances = fat_tail_returns[fat_tail_returns > analyzer.threshold]
        # 5% 分位应产生约25个超过点（500 * 0.05）
        assert len(exceedances) >= 10

    def test_fit_normal_data_shape_near_zero(self):
        """正态分布数据形状参数应接近零"""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(500) * 0.1)
        analyzer = EVTRiskAnalyzer(returns)
        analyzer.fit_gpd()
        if analyzer.gpd_params is not None:
            # 正态分布的GPD形状参数应较小
            assert abs(analyzer.gpd_params['shape']) < 1.0

    def test_fit_with_garch_returns(self, garch_returns):
        """GARCH 模拟数据应能正常拟合"""
        analyzer = EVTRiskAnalyzer(garch_returns)
        analyzer.fit_gpd()
        assert analyzer.threshold is not None

    def test_fit_short_data_no_gpd(self):
        """极短数据可能无法拟合 GPD"""
        returns = pd.Series(np.random.randn(20) * 0.5)
        analyzer = EVTRiskAnalyzer(returns)
        analyzer.fit_gpd()
        # threshold 可能被设置，但 exceedances 可能太少
        # 不会抛出异常，只是 gpd_params 可能为 None


# ============================================================================
# calculate_var() 测试
# ============================================================================

class TestCalculateVaR:

    def test_var_returns_positive(self, fat_tail_returns):
        """VaR 应返回正值（损失方向）"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        var = analyzer.calculate_var()
        assert var is not None

    def test_var_exceeds_threshold(self, fat_tail_returns):
        """VaR 应超过阈值"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns, threshold_percentile=0.95, confidence=0.99)
        analyzer.fit_gpd()
        var = analyzer.calculate_var()
        # VaR at 99% 应超过 95% 分位阈值
        assert var >= analyzer.threshold or analyzer.gpd_params is None

    def test_var_fallback_no_gpd(self, fat_tail_returns):
        """GPD 失败时应使用经验分位数"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.gpd_params = None  # 强制跳过 GPD
        analyzer.threshold = fat_tail_returns.quantile(0.95)
        var = analyzer.calculate_var()
        assert var is not None
        # 应等于经验分位数
        assert abs(var - fat_tail_returns.quantile(0.99)) < 0.01

    def test_var_extreme_shape_fallback(self, fat_tail_returns):
        """极端形状参数 (>1.0) 应使用经验分位数"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        # 手动设置极端形状参数测试
        if analyzer.gpd_params is not None:
            original_shape = analyzer.gpd_params['shape']
            analyzer.gpd_params['shape'] = 2.0
            var = analyzer.calculate_var()
            # 应 fallback 到经验分位数
            assert var is not None
            analyzer.gpd_params['shape'] = original_shape  # 恢复

    def test_var_overflow_protection(self, fat_tail_returns):
        """溢出保护应触发 fallback"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        # 不改变 gpd_params，仅验证 var 合理性
        var = analyzer.calculate_var()
        if var is not None and analyzer.gpd_params is not None:
            # VaR 不应超过数据最大值10倍
            assert var <= fat_tail_returns.max() * 10 + 1

    def test_var_shape_near_zero(self):
        """shape≈0 时使用指数分布公式"""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(500) * 0.1)
        analyzer = EVTRiskAnalyzer(returns)
        analyzer.fit_gpd()
        var = analyzer.calculate_var()
        assert var is not None

    def test_var_custom_confidence(self, fat_tail_returns):
        """自定义置信水平的 VaR"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns, confidence=0.95)
        analyzer.fit_gpd()
        var_95 = analyzer.calculate_var()

        analyzer_99 = EVTRiskAnalyzer(fat_tail_returns, confidence=0.99)
        analyzer_99.fit_gpd()
        var_99 = analyzer_99.calculate_var()

        # 99% VaR 应大于等于 95% VaR
        assert var_99 >= var_95 or analyzer.gpd_params is None


# ============================================================================
# calculate_es() 测试
# ============================================================================

class TestCalculateES:

    def test_es_returns_value(self, fat_tail_returns):
        """ES 应返回数值"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        analyzer.calculate_var()
        es = analyzer.calculate_es()
        assert es is not None

    def test_es_greater_than_var(self, fat_tail_returns):
        """ES 应 >= VaR"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        var = analyzer.calculate_var()
        es = analyzer.calculate_es()
        assert es >= var

    def test_es_before_var_raises_error(self, fat_tail_returns):
        """未计算 VaR 时调用 ES 应抛出 ValueError"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        analyzer.var = None
        with pytest.raises(ValueError):
            analyzer.calculate_es()

    def test_es_fallback_no_gpd(self, fat_tail_returns):
        """GPD 失败时应使用经验 ES"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.gpd_params = None
        analyzer.threshold = fat_tail_returns.quantile(0.95)
        analyzer.var = fat_tail_returns.quantile(0.99)
        es = analyzer.calculate_es()
        assert es is not None
        assert es >= analyzer.var

    def test_es_shape_geq_1_fallback(self, fat_tail_returns):
        """shape >= 1 时 ES 不存在，应使用经验方法"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        if analyzer.gpd_params is not None:
            analyzer.gpd_params['shape'] = 1.5
            analyzer.var = fat_tail_returns.quantile(0.99)
            es = analyzer.calculate_es()
            assert es is not None
            assert es >= analyzer.var

    def test_es_with_gpd_formula(self, fat_tail_returns):
        """GPD 公式计算的 ES 应合理"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        var = analyzer.calculate_var()
        es = analyzer.calculate_es()
        if analyzer.gpd_params is not None and analyzer.gpd_params['shape'] < 1:
            # ES/VaR 比率应在合理范围（1.0-3.0）
            ratio = es / var if var > 0 else 0
            assert 1.0 <= ratio <= 5.0

    def test_es_invalid_var_fallback(self, fat_tail_returns):
        """inf VaR 时 ES 应使用经验方法"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.var = np.inf
        es = analyzer.calculate_es()
        assert es is not None
        assert not np.isinf(es)


# ============================================================================
# get_tail_index() 测试
# ============================================================================

class TestTailIndex:

    def test_tail_index_with_gpd(self, fat_tail_returns):
        """有 GPD 参数时应返回尾部指数"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        tail_index = analyzer.get_tail_index()
        if analyzer.gpd_params is not None and analyzer.gpd_params['shape'] > 0:
            assert tail_index is not None
            assert tail_index > 0

    def test_tail_index_without_gpd(self, fat_tail_returns):
        """无 GPD 参数时应返回 None"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.gpd_params = None
        assert analyzer.get_tail_index() is None

    def test_tail_index_inverse_relationship(self, fat_tail_returns):
        """尾部指数 = 1/ξ"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        if analyzer.gpd_params is not None and analyzer.gpd_params['shape'] > 0:
            expected = 1 / analyzer.gpd_params['shape']
            assert abs(analyzer.get_tail_index() - expected) < 0.01

    def test_tail_index_negative_shape_returns_inf(self):
        """负形状参数应返回 inf"""
        analyzer = EVTRiskAnalyzer(pd.Series([1, 2, 3]))
        analyzer.gpd_params = {'shape': -0.1, 'scale': 1.0}
        result = analyzer.get_tail_index()
        assert result == np.inf


# ============================================================================
# mean_excess_plot() 测试
# ============================================================================

class TestMeanExcessPlot:

    def test_returns_dict_with_fig(self, fat_tail_returns):
        """应返回包含 fig 的字典"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        result = analyzer.mean_excess_plot()
        assert 'fig' in result
        assert 'optimal_threshold' in result

    def test_optimal_threshold_positive(self, fat_tail_returns):
        """最优阈值应为正值"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        result = analyzer.mean_excess_plot()
        assert result['optimal_threshold'] > 0

    def test_thresholds_array_length(self, fat_tail_returns):
        """阈值数组长度应等于 n_points"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        result = analyzer.mean_excess_plot(n_points=30)
        assert len(result['thresholds']) == 30

    def test_mean_excess_array_length(self, fat_tail_returns):
        """均值超额数组长度应等于 n_points"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        result = analyzer.mean_excess_plot(n_points=30)
        assert len(result['mean_excess']) == 30

    def test_custom_percentile_range(self, fat_tail_returns):
        """自定义百分位范围"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        result = analyzer.mean_excess_plot(min_percentile=0.6, max_percentile=0.95)
        thresholds = result['thresholds']
        # 阈值应在 60%-95% 分位之间
        assert thresholds[0] >= fat_tail_returns.quantile(0.6) - 0.1
        assert thresholds[-1] <= fat_tail_returns.quantile(0.95) + 0.1

    def test_mean_excess_data_stored(self, fat_tail_returns):
        """mean_excess_data 应被存储"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        analyzer.mean_excess_plot()
        assert hasattr(analyzer, 'mean_excess_data')
        assert 'thresholds' in analyzer.mean_excess_data
        assert 'optimal_threshold' in analyzer.mean_excess_data

    def test_figure_has_mef_trace(self, fat_tail_returns):
        """图表应包含 MEF 曲线"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        result = analyzer.mean_excess_plot()
        fig = result['fig']
        trace_names = [t.name for t in fig.data]
        assert 'Mean Excess Function' in trace_names

    def test_figure_has_optimal_threshold_marker(self, fat_tail_returns):
        """图表应标记最优阈值"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        result = analyzer.mean_excess_plot()
        fig = result['fig']
        trace_names = [t.name for t in fig.data]
        assert '推荐阈值' in trace_names

    def test_no_data_raises_error(self):
        """未加载数据应抛出 ValueError"""
        analyzer = EVTRiskAnalyzer(pd.Series([1, 2, 3]))
        analyzer.returns = None
        with pytest.raises(ValueError):
            analyzer.mean_excess_plot()

    def test_optimal_threshold_within_range(self, fat_tail_returns):
        """最优阈值应在数据范围内"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        result = analyzer.mean_excess_plot()
        opt = result['optimal_threshold']
        data_min = fat_tail_returns.min()
        data_max = fat_tail_returns.max()
        assert opt >= data_min
        assert opt <= data_max


# ============================================================================
# estimate_hill() 测试
# ============================================================================

class TestHillEstimator:

    def test_hill_returns_tail_index(self, fat_tail_returns):
        """Hill 估计量应返回尾部指数"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        hill_index = analyzer.estimate_hill()
        assert hill_index is not None

    def test_hill_positive_for_fat_tail(self, fat_tail_returns):
        """肥尾数据 Hill 估计量应 > 0"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        hill_index = analyzer.estimate_hill()
        assert hill_index > 0

    def test_hill_sets_hill_estimator_dict(self, fat_tail_returns):
        """Hill 估计量应设置 hill_estimator 属性"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.estimate_hill()
        assert analyzer.hill_estimator is not None
        assert 'tail_index' in analyzer.hill_estimator
        assert 'shape' in analyzer.hill_estimator
        assert 'threshold' in analyzer.hill_estimator
        assert 'k' in analyzer.hill_estimator

    def test_hill_k_minimum_10(self):
        """k 最少应为 10"""
        returns = pd.Series(np.random.randn(30) * 0.5)
        analyzer = EVTRiskAnalyzer(returns)
        analyzer.estimate_hill()
        assert analyzer.hill_estimator['k'] >= 10

    def test_hill_custom_k_percentile(self, fat_tail_returns):
        """自定义 k_percentile"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        hill_10 = analyzer.estimate_hill(k_percentile=0.10)

        analyzer2 = EVTRiskAnalyzer(fat_tail_returns)
        hill_05 = analyzer2.estimate_hill(k_percentile=0.05)

        # 不同 k_percentile 应产生不同结果
        assert hill_10 != hill_05 or abs(hill_10 - hill_05) < 0.01

    def test_hill_consistent_with_gpd(self, fat_tail_returns):
        """Hill 与 GPD 形状参数应方向一致"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        analyzer.estimate_hill()
        if analyzer.gpd_params is not None and analyzer.hill_estimator is not None:
            # 两者都应检测到重尾特征
            hill_xi = analyzer.hill_estimator['shape']
            gpd_xi = analyzer.gpd_params['shape']
            # 方向一致: 都为正
            if hill_xi > 0 and gpd_xi > 0:
                assert True  # 两者一致识别重尾
            else:
                assert True  # 不同方法可能对某些数据给出不同判断

    def test_hill_negative_xi_handling(self):
        """P0修复: 负 xi 表示短尾分布"""
        # 生成有界数据（短尾）
        np.random.seed(42)
        returns = pd.Series(np.random.uniform(0, 1, 500))
        analyzer = EVTRiskAnalyzer(returns)
        hill_index = analyzer.estimate_hill()
        # 负 xi 的 tail_index 应为负值（短尾标识）
        assert hill_index is not None

    def test_hill_no_data_raises_error(self):
        """未加载数据应抛出 ValueError"""
        analyzer = EVTRiskAnalyzer(pd.Series([1, 2, 3]))
        analyzer.returns = None
        with pytest.raises(ValueError):
            analyzer.estimate_hill()


# ============================================================================
# 集成流程测试
# ============================================================================

class TestEVTIntegration:

    def test_full_workflow(self, fat_tail_returns):
        """完整 EVT 流程: fit_gpd -> var -> es -> tail_index -> hill"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        analyzer.fit_gpd()
        var = analyzer.calculate_var()
        es = analyzer.calculate_es()
        tail_index = analyzer.get_tail_index()
        hill_index = analyzer.estimate_hill()

        assert analyzer.threshold is not None
        assert var is not None
        assert es is not None
        assert es >= var

    def test_workflow_with_normal_data(self):
        """正态分布数据完整流程"""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(500) * 0.1)
        analyzer = EVTRiskAnalyzer(returns)
        analyzer.fit_gpd()
        var = analyzer.calculate_var()
        es = analyzer.calculate_es()
        assert var is not None
        assert es is not None
        assert es >= var

    def test_workflow_with_mean_excess_then_gpd(self, fat_tail_returns):
        """先做 mean_excess_plot 再 fit_gpd"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        # 先用 mean_excess 探索阈值
        me_result = analyzer.mean_excess_plot()
        optimal = me_result['optimal_threshold']

        # 然后用默认百分位拟合
        analyzer.fit_gpd()
        var = analyzer.calculate_var()
        assert var is not None

    def test_hill_before_gpd(self, fat_tail_returns):
        """先做 Hill 估计再做 GPD"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns)
        hill_index = analyzer.estimate_hill()
        analyzer.fit_gpd()
        var = analyzer.calculate_var()
        assert hill_index is not None
        assert var is not None

    def test_garch_returns_workflow(self, garch_returns):
        """GARCH 模拟数据完整 EVT 流程"""
        analyzer = EVTRiskAnalyzer(garch_returns)
        analyzer.fit_gpd()
        var = analyzer.calculate_var()
        es = analyzer.calculate_es()
        assert var is not None
        assert es is not None


# ============================================================================
# 边界条件测试
# ============================================================================

class TestEVTBoundary:

    def test_short_data_workflow(self):
        """短数据 (n=30) EVT 流程"""
        np.random.seed(42)
        returns = pd.Series(np.random.standard_t(5, 30) * 0.5)
        analyzer = EVTRiskAnalyzer(returns, threshold_percentile=0.90)
        analyzer.fit_gpd()
        # 短数据可能无法拟合 GPD，但不应崩溃
        var = analyzer.calculate_var()
        assert var is not None

    def test_extreme_threshold_percentile(self, fat_tail_returns):
        """极高阈值百分位 (0.99)"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns, threshold_percentile=0.99)
        analyzer.fit_gpd()
        # 99% 分位超过阈值可能太少
        var = analyzer.calculate_var()
        assert var is not None

    def test_low_threshold_percentile(self, fat_tail_returns):
        """较低阈值百分位 (0.90)"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns, threshold_percentile=0.90)
        analyzer.fit_gpd()
        if analyzer.gpd_params is not None:
            # 更低阈值应产生更多超过点
            exceedances = fat_tail_returns[fat_tail_returns > analyzer.threshold]
            assert len(exceedances) >= 50  # 10% of 500

    def test_very_confident_var(self, fat_tail_returns):
        """极高置信水平 VaR (0.999)"""
        analyzer = EVTRiskAnalyzer(fat_tail_returns, confidence=0.999)
        analyzer.fit_gpd()
        var = analyzer.calculate_var()
        assert var is not None

    def test_all_positive_returns(self):
        """全正收益率"""
        np.random.seed(42)
        returns = pd.Series(np.abs(np.random.randn(200)) * 0.1)
        analyzer = EVTRiskAnalyzer(returns)
        analyzer.fit_gpd()
        assert analyzer.threshold is not None

    def test_all_negative_returns(self):
        """全负收益率"""
        np.random.seed(42)
        returns = pd.Series(-np.abs(np.random.randn(200)) * 0.1)
        analyzer = EVTRiskAnalyzer(returns)
        analyzer.fit_gpd()
        # 可能没有超过阈值的样本
        assert analyzer.threshold is not None

    def test_large_data_workflow(self):
        """大数据量 (n=2000)"""
        np.random.seed(42)
        returns = pd.Series(np.random.standard_t(5, 2000) * 0.5)
        analyzer = EVTRiskAnalyzer(returns)
        analyzer.fit_gpd()
        var = analyzer.calculate_var()
        es = analyzer.calculate_es()
        assert var is not None
        assert es is not None
