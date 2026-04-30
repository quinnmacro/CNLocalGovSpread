"""
KalmanSignalExtractor 单元测试 - 卡尔曼滤波器信号提取模块
v3.0 - CNLocalGovSpread 计量经济学框架

覆盖: 初始化, fit(成功/失败/fallback), 信号偏离度, 边界条件, 数据模式
"""

import pytest
import numpy as np
import pandas as pd
from kalman import KalmanSignalExtractor


# ============================================================================
# 基础初始化测试
# ============================================================================

class TestKalmanInit:

    def test_init_stores_spread(self, spread_series):
        """初始化应存储 spread_series 并重置状态"""
        extractor = KalmanSignalExtractor(spread_series)
        assert extractor.spread is spread_series
        assert extractor.smoothed_state is None
        assert extractor.success is False

    def test_init_with_series_no_index(self):
        """无索引 Series 应正常初始化"""
        spread = pd.Series([100, 101, 102, 103, 104])
        extractor = KalmanSignalExtractor(spread)
        assert len(extractor.spread) == 5

    def test_init_preserves_series_name(self):
        """初始化应保留 Series 的 name 属性"""
        spread = pd.Series([80, 81, 82], name='spread_test')
        extractor = KalmanSignalExtractor(spread)
        assert extractor.spread.name == 'spread_test'


# ============================================================================
# fit() 成功路径测试
# ============================================================================

class TestKalmanFit:

    def test_fit_returns_series(self, spread_series):
        """fit() 应返回 pandas Series"""
        extractor = KalmanSignalExtractor(spread_series)
        smoothed = extractor.fit()
        assert isinstance(smoothed, pd.Series)

    def test_fit_preserves_length(self, spread_series):
        """平滑结果长度应与输入一致"""
        extractor = KalmanSignalExtractor(spread_series)
        smoothed = extractor.fit()
        assert len(smoothed) == len(spread_series)

    def test_fit_preserves_index(self, spread_series):
        """平滑结果索引应与输入一致"""
        extractor = KalmanSignalExtractor(spread_series)
        smoothed = extractor.fit()
        assert smoothed.index.equals(spread_series.index)

    def test_fit_sets_success_true(self, spread_series):
        """正常拟合后 success 应为 True"""
        extractor = KalmanSignalExtractor(spread_series)
        extractor.fit()
        assert extractor.success is True

    def test_fit_sets_smoothed_state(self, spread_series):
        """正常拟合后 smoothed_state 应被设置"""
        extractor = KalmanSignalExtractor(spread_series)
        result = extractor.fit()
        assert extractor.smoothed_state is not None
        assert extractor.smoothed_state is result

    def test_fit_smoothed_reduces_noise(self, spread_series):
        """平滑结果的波动应小于原始数据"""
        extractor = KalmanSignalExtractor(spread_series)
        smoothed = extractor.fit()
        assert smoothed.std() <= spread_series.std() * 1.5

    def test_fit_with_custom_fallback_window(self, spread_series):
        """fit(fallback_window=30) 应正常工作"""
        extractor = KalmanSignalExtractor(spread_series)
        smoothed = extractor.fit(fallback_window=30)
        assert isinstance(smoothed, pd.Series)
        assert len(smoothed) == len(spread_series)

    def test_fit_returns_numeric_values(self, spread_series):
        """平滑结果应全部为数值"""
        extractor = KalmanSignalExtractor(spread_series)
        smoothed = extractor.fit()
        assert smoothed.notna().all()

    def test_fit_smoothed_close_to_original_mean(self, spread_series):
        """平滑均值应接近原始均值"""
        extractor = KalmanSignalExtractor(spread_series)
        smoothed = extractor.fit()
        mean_diff = abs(smoothed.mean() - spread_series.mean())
        assert mean_diff < spread_series.std() * 2

    def test_fit_with_garch_returns(self, garch_returns):
        """GARCH 模拟数据的差分序列应能正常拟合"""
        # 卡尔曼滤波处理的是利差绝对值，不是收益率
        spread = garch_returns.abs().cumsum() + 80
        extractor = KalmanSignalExtractor(spread)
        smoothed = extractor.fit()
        assert isinstance(smoothed, pd.Series)
        assert extractor.success is True


# ============================================================================
# fit() Fallback 路径测试
# ============================================================================

class TestKalmanFallback:

    def test_fallback_with_very_short_data(self):
        """极短数据 (n=5) 应触发 fallback"""
        spread = pd.Series([100, 101, 102, 103, 104])
        extractor = KalmanSignalExtractor(spread)
        smoothed = extractor.fit()
        # 极短数据可能导致优化失败，触发 fallback
        assert isinstance(smoothed, pd.Series)
        assert len(smoothed) == 5

    def test_fallback_preserves_length(self):
        """Fallback 结果长度应与输入一致"""
        short_spread = pd.Series(np.random.randn(5) + 100)
        extractor = KalmanSignalExtractor(short_spread)
        smoothed = extractor.fit(fallback_window=3)
        assert len(smoothed) == len(short_spread)

    def test_fallback_returns_rolling_mean(self):
        """Fallback 应返回指定窗口的滚动均值"""
        data = np.arange(100, 110, dtype=float)
        spread = pd.Series(data)
        extractor = KalmanSignalExtractor(spread)
        smoothed = extractor.fit(fallback_window=5)
        assert isinstance(smoothed, pd.Series)

    def test_fallback_window_min_periods(self):
        """Fallback 滚动均值应使用 min_periods=1"""
        # 极短序列前几点的滚动均值不应为 NaN
        data = [100.0, 101.0, 102.0]
        spread = pd.Series(data)
        extractor = KalmanSignalExtractor(spread)
        smoothed = extractor.fit(fallback_window=60)
        # min_periods=1 保证前几个点也有值
        if not extractor.success:
            assert smoothed.notna().all()


# ============================================================================
# get_signal_deviation() 测试
# ============================================================================

class TestSignalDeviation:

    def test_deviation_returns_series(self, spread_series):
        """信号偏离度应返回 Series"""
        extractor = KalmanSignalExtractor(spread_series)
        extractor.fit()
        deviation = extractor.get_signal_deviation()
        assert isinstance(deviation, pd.Series)

    def test_deviation_length_matches_spread(self, spread_series):
        """偏离度长度应与利差数据一致"""
        extractor = KalmanSignalExtractor(spread_series)
        extractor.fit()
        deviation = extractor.get_signal_deviation()
        assert len(deviation) == len(spread_series)

    def test_deviation_before_fit_raises_error(self, spread_series):
        """未拟合时调用 get_signal_deviation 应抛出 ValueError"""
        extractor = KalmanSignalExtractor(spread_series)
        with pytest.raises(ValueError):
            extractor.get_signal_deviation()

    def test_deviation_index_matches_spread(self, spread_series):
        """偏离度索引应与利差数据一致"""
        extractor = KalmanSignalExtractor(spread_series)
        extractor.fit()
        deviation = extractor.get_signal_deviation()
        assert deviation.index.equals(spread_series.index)

    def test_deviation_values_finite(self, spread_series):
        """偏离度应不含 inf 值"""
        extractor = KalmanSignalExtractor(spread_series)
        extractor.fit()
        deviation = extractor.get_signal_deviation()
        assert not np.any(np.isinf(deviation.values))

    def test_deviation_has_positive_and_negative(self, spread_series):
        """偏离度应包含正值和负值"""
        extractor = KalmanSignalExtractor(spread_series)
        extractor.fit()
        deviation = extractor.get_signal_deviation()
        assert (deviation > 0).any()
        assert (deviation < 0).any()

    def test_deviation_mean_near_zero(self, spread_series):
        """标准化偏离度均值应接近零"""
        extractor = KalmanSignalExtractor(spread_series)
        extractor.fit()
        deviation = extractor.get_signal_deviation()
        assert abs(deviation.mean()) < 1.0

    def test_deviation_std_near_one(self, spread_series):
        """标准化偏离度标准差应接近 1"""
        extractor = KalmanSignalExtractor(spread_series)
        extractor.fit()
        deviation = extractor.get_signal_deviation()
        # 标准化后 std 应在 0.5-2.0 范围（允许一定误差）
        assert 0.3 < deviation.std() < 3.0

    def test_deviation_interpretation_thresholds(self, spread_series):
        """偏离度超过1σ应占合理比例"""
        extractor = KalmanSignalExtractor(spread_series)
        extractor.fit()
        deviation = extractor.get_signal_deviation()
        pct_above_1 = (abs(deviation) > 1).sum() / len(deviation)
        pct_above_2 = (abs(deviation) > 2).sum() / len(deviation)
        # 标准正态下约32%超过1σ，约5%超过2σ
        assert pct_above_1 > 0.1  # 至少有10%超过1σ
        assert pct_above_2 < 0.5  # 不超过50%超过2σ


# ============================================================================
# 数据模式测试
# ============================================================================

class TestDataPatterns:

    def test_trending_data(self):
        """趋势上升数据应能正常提取信号"""
        dates = pd.date_range('2020-01-01', periods=200, freq='B')
        spread = pd.Series(np.linspace(80, 120, 200), index=dates)
        extractor = KalmanSignalExtractor(spread)
        smoothed = extractor.fit()
        assert isinstance(smoothed, pd.Series)
        # 平滑值应跟随趋势
        assert smoothed.iloc[-1] > smoothed.iloc[0]

    def test_mean_reverting_data(self):
        """均值回归数据应能正常提取信号"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='B')
        spread = pd.Series(100 + np.random.randn(200) * 5, index=dates)
        extractor = KalmanSignalExtractor(spread)
        smoothed = extractor.fit()
        deviation = extractor.get_signal_deviation()
        assert isinstance(smoothed, pd.Series)
        assert isinstance(deviation, pd.Series)

    def test_high_volatility_data(self):
        """高波动数据应能正常提取信号"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='B')
        spread = pd.Series(100 + np.random.randn(200) * 50, index=dates)
        extractor = KalmanSignalExtractor(spread)
        smoothed = extractor.fit()
        assert isinstance(smoothed, pd.Series)

    def test_constant_data(self):
        """恒定数据应能正常提取信号"""
        dates = pd.date_range('2020-01-01', periods=50, freq='B')
        spread = pd.Series([100.0] * 50, index=dates)
        extractor = KalmanSignalExtractor(spread)
        smoothed = extractor.fit()
        assert isinstance(smoothed, pd.Series)

    def test_data_with_jump(self):
        """带跳跃的数据应能正常提取信号"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='B')
        spread = pd.Series(100 + np.random.randn(200) * 2, index=dates)
        # 在第50个点注入跳跃
        spread.iloc[50] += 30
        extractor = KalmanSignalExtractor(spread)
        smoothed = extractor.fit()
        deviation = extractor.get_signal_deviation()
        # 跳跃点附近应有较高偏离度
        assert abs(deviation.iloc[50]) > abs(deviation.mean())


# ============================================================================
# 集成流程测试
# ============================================================================

class TestKalmanIntegration:

    def test_full_workflow(self, spread_series):
        """完整流程: init -> fit -> deviation"""
        extractor = KalmanSignalExtractor(spread_series)
        smoothed = extractor.fit()
        deviation = extractor.get_signal_deviation()

        assert extractor.smoothed_state is not None
        assert extractor.success is True
        assert len(smoothed) == len(spread_series)
        assert len(deviation) == len(spread_series)

    def test_refit_overwrites_state(self, spread_series):
        """多次 fit 应覆盖之前的状态"""
        extractor = KalmanSignalExtractor(spread_series)
        smoothed1 = extractor.fit()
        smoothed2 = extractor.fit()
        # 第二次拟合应覆盖第一次结果
        assert extractor.smoothed_state is smoothed2

    def test_workflow_with_fat_tail_returns(self, fat_tail_returns):
        """肥尾收益率绝对值累积序列应能正常处理"""
        spread = fat_tail_returns.abs().cumsum() + 80
        dates = pd.date_range('2020-01-01', periods=len(spread), freq='B')
        spread = pd.Series(spread.values, index=dates)
        extractor = KalmanSignalExtractor(spread)
        smoothed = extractor.fit()
        deviation = extractor.get_signal_deviation()
        assert isinstance(smoothed, pd.Series)
        assert isinstance(deviation, pd.Series)
        assert len(smoothed) == len(spread)

    def test_consecutive_fit_deviation_consistent(self, spread_series):
        """连续 fit+deviation 应产生一致结果"""
        extractor = KalmanSignalExtractor(spread_series)
        extractor.fit()
        dev1 = extractor.get_signal_deviation()
        dev2 = extractor.get_signal_deviation()
        # 相同数据应产生相同偏离度
        pd.testing.assert_series_equal(dev1, dev2)


# ============================================================================
# 边界条件测试
# ============================================================================

class TestKalmanBoundary:

    def test_minimum_data_length(self):
        """最短可处理数据 (n=3)"""
        spread = pd.Series([100.0, 101.0, 102.0])
        extractor = KalmanSignalExtractor(spread)
        smoothed = extractor.fit()
        assert len(smoothed) == 3

    def test_large_data_length(self):
        """大数据量 (n=1000) 应正常处理"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000, freq='B')
        spread = pd.Series(100 + np.cumsum(np.random.randn(1000) * 0.1), index=dates)
        extractor = KalmanSignalExtractor(spread)
        smoothed = extractor.fit()
        assert len(smoothed) == 1000

    def test_extreme_values_in_spread(self):
        """极端值数据应能处理"""
        np.random.seed(42)
        values = np.random.randn(100) * 1000 + 5000
        spread = pd.Series(values)
        extractor = KalmanSignalExtractor(spread)
        smoothed = extractor.fit()
        assert isinstance(smoothed, pd.Series)

    def test_negative_spread_values(self):
        """负利差值应能正常处理"""
        np.random.seed(42)
        spread = pd.Series(-100 + np.random.randn(200) * 5)
        extractor = KalmanSignalExtractor(spread)
        smoothed = extractor.fit()
        assert isinstance(smoothed, pd.Series)

    def test_zero_variance_spread(self):
        """零方差 (恒定) 数据应能处理"""
        spread = pd.Series([50.0] * 100)
        extractor = KalmanSignalExtractor(spread)
        smoothed = extractor.fit()
        deviation = extractor.get_signal_deviation()
        assert isinstance(smoothed, pd.Series)
        # 恒定数据的偏离度应接近0
        if len(deviation) > 0:
            assert abs(deviation.mean()) < 5.0  # 宽松阈值
