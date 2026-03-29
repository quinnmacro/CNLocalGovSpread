"""
单元测试 - 测试所有核心模块
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_engine import DataEngine
from volatility import VolatilityModeler
from kalman import KalmanSignalExtractor
from evt import EVTRiskAnalyzer


# ============================================================================
# 测试配置
# ============================================================================

@pytest.fixture
def test_config():
    """测试用配置"""
    return {
        'SOURCE': 'MOCK',
        'TICKER': 'TEST',
        'START_DATE': '2020-01-01',
        'END_DATE': '2022-12-31',
        'MAD_THRESHOLD': 5.0,
        'GARCH_P': 1,
        'GARCH_Q': 1,
        'VaR_CONFIDENCE': 0.99,
        'EVT_THRESHOLD_PERCENTILE': 0.95
    }


# ============================================================================
# DataEngine 测试
# ============================================================================

class TestDataEngine:
    """DataEngine 类测试"""

    def test_load_data_returns_dataframe(self, test_config):
        """测试 load_data 返回 DataFrame"""
        engine = DataEngine(test_config)
        data = engine.load_data()
        assert isinstance(data, pd.DataFrame)
        assert 'spread' in data.columns

    def test_load_data_date_range(self, test_config):
        """测试数据日期范围"""
        engine = DataEngine(test_config)
        data = engine.load_data()
        assert data.index[0] >= pd.Timestamp(test_config['START_DATE'])
        assert data.index[-1] <= pd.Timestamp(test_config['END_DATE'])

    def test_clean_data_removes_outliers(self, test_config):
        """测试异常值处理"""
        engine = DataEngine(test_config)
        engine.load_data()
        clean = engine.clean_data()
        assert isinstance(clean, pd.DataFrame)
        assert len(clean) > 0

    def test_get_returns_calculates_diff(self, test_config):
        """测试收益率计算"""
        engine = DataEngine(test_config)
        engine.load_data()
        engine.clean_data()
        returns = engine.get_returns()
        assert isinstance(returns, pd.Series)
        assert len(returns) == len(engine._clean_data) - 1  # 差分会少一个

    def test_clean_data_before_load_raises_error(self, test_config):
        """测试未加载数据时调用 clean_data 抛出异常"""
        engine = DataEngine(test_config)
        with pytest.raises(ValueError):
            engine.clean_data()


# ============================================================================
# VolatilityModeler 测试
# ============================================================================

class TestVolatilityModeler:
    """VolatilityModeler 类测试"""

    @pytest.fixture
    def sample_returns(self):
        """生成测试用收益率数据"""
        np.random.seed(42)
        return pd.Series(np.random.randn(500) * 0.1)

    def test_tournament_returns_winner(self, sample_returns):
        """测试锦标赛返回获胜模型名称"""
        modeler = VolatilityModeler(sample_returns)
        winner = modeler.run_tournament()
        assert winner in ['GARCH', 'EGARCH', 'GJR-GARCH']

    def test_tournament_populates_ic_scores(self, sample_returns):
        """测试锦标赛填充 IC 分数"""
        modeler = VolatilityModeler(sample_returns)
        modeler.run_tournament()
        assert 'GARCH' in modeler.ic_scores
        assert 'EGARCH' in modeler.ic_scores
        assert 'GJR-GARCH' in modeler.ic_scores
        for scores in modeler.ic_scores.values():
            assert 'AIC' in scores
            assert 'BIC' in scores

    def test_get_conditional_volatility(self, sample_returns):
        """测试获取条件波动率"""
        modeler = VolatilityModeler(sample_returns)
        winner = modeler.run_tournament()
        vol = modeler.get_conditional_volatility(winner)
        assert isinstance(vol, pd.Series)
        assert len(vol) > 0
        assert (vol > 0).all()  # 波动率必须为正


# ============================================================================
# KalmanSignalExtractor 测试
# ============================================================================

class TestKalmanSignalExtractor:
    """KalmanSignalExtractor 类测试"""

    @pytest.fixture
    def sample_spread(self):
        """生成测试用利差数据"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='B')
        spread = 100 + np.cumsum(np.random.randn(500) * 0.1)
        return pd.Series(spread, index=dates)

    def test_fit_returns_series(self, sample_spread):
        """测试 fit 返回 Series"""
        extractor = KalmanSignalExtractor(sample_spread)
        smoothed = extractor.fit()
        assert isinstance(smoothed, pd.Series)
        assert len(smoothed) == len(sample_spread)

    def test_smoothed_state_preserves_index(self, sample_spread):
        """测试平滑结果保持索引"""
        extractor = KalmanSignalExtractor(sample_spread)
        smoothed = extractor.fit()
        assert smoothed.index.equals(sample_spread.index)

    def test_get_signal_deviation(self, sample_spread):
        """测试信号偏离度计算"""
        extractor = KalmanSignalExtractor(sample_spread)
        extractor.fit()
        deviation = extractor.get_signal_deviation()
        assert isinstance(deviation, pd.Series)
        assert len(deviation) == len(sample_spread)

    def test_deviation_before_fit_raises_error(self, sample_spread):
        """测试未拟合时调用 get_signal_deviation 抛出异常"""
        extractor = KalmanSignalExtractor(sample_spread)
        with pytest.raises(ValueError):
            extractor.get_signal_deviation()


# ============================================================================
# EVTRiskAnalyzer 测试
# ============================================================================

class TestEVTRiskAnalyzer:
    """EVTRiskAnalyzer 类测试"""

    @pytest.fixture
    def sample_returns(self):
        """生成测试用收益率数据（带肥尾）"""
        np.random.seed(42)
        return pd.Series(np.random.standard_t(5, 500) * 0.5)

    def test_fit_gpd_calculates_threshold(self, sample_returns):
        """测试 GPD 拟合计算阈值"""
        analyzer = EVTRiskAnalyzer(sample_returns)
        analyzer.fit_gpd()
        assert analyzer.threshold is not None
        assert analyzer.threshold > sample_returns.quantile(0.90)

    def test_fit_gpd_estimates_params(self, sample_returns):
        """测试 GPD 参数估计"""
        analyzer = EVTRiskAnalyzer(sample_returns)
        analyzer.fit_gpd()
        if analyzer.gpd_params is not None:
            assert 'shape' in analyzer.gpd_params
            assert 'scale' in analyzer.gpd_params
            assert analyzer.gpd_params['shape'] > 0  # 金融数据通常重尾

    def test_calculate_var(self, sample_returns):
        """测试 VaR 计算"""
        analyzer = EVTRiskAnalyzer(sample_returns)
        analyzer.fit_gpd()
        var = analyzer.calculate_var()
        assert var is not None
        assert var > 0  # VaR 应该是正数

    def test_get_tail_index(self, sample_returns):
        """测试尾部指数"""
        analyzer = EVTRiskAnalyzer(sample_returns)
        analyzer.fit_gpd()
        tail_index = analyzer.get_tail_index()
        if tail_index is not None:
            assert tail_index > 0


# ============================================================================
# 集成测试
# ============================================================================

class TestIntegration:
    """集成测试 - 测试完整工作流"""

    def test_full_workflow(self, test_config):
        """测试完整分析流程"""
        # 1. 数据加载
        engine = DataEngine(test_config)
        engine.load_data()
        clean_data = engine.clean_data()
        returns = engine.get_returns()

        # 2. GARCH 锦标赛
        vol_modeler = VolatilityModeler(returns)
        winner = vol_modeler.run_tournament()
        assert winner is not None

        # 3. 卡尔曼滤波
        kalman = KalmanSignalExtractor(clean_data['spread'])
        smoothed = kalman.fit()
        deviation = kalman.get_signal_deviation()
        assert smoothed is not None
        assert deviation is not None

        # 4. EVT 分析
        evt = EVTRiskAnalyzer(returns)
        evt.fit_gpd()
        var = evt.calculate_var()
        assert var is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
