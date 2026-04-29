"""
ML波动率模型对比模块单元测试
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_volatility import MLVolatilityModeler


class TestMLVolatilityModeler:
    """MLVolatilityModeler 类测试"""

    @pytest.fixture
    def sample_returns(self):
        """生成测试用收益率数据（带肥尾和波动率聚集）"""
        np.random.seed(42)
        n = 500
        # 模拟GARCH过程生成数据
        returns = np.zeros(n)
        vol = np.zeros(n)
        vol[0] = 0.5
        for t in range(1, n):
            vol[t] = np.sqrt(0.1 + 0.15 * returns[t-1]**2 + 0.75 * vol[t-1]**2)
            returns[t] = np.random.standard_t(5) * vol[t]
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        return pd.Series(returns, index=dates)

    def test_init(self, sample_returns):
        """测试初始化"""
        modeler = MLVolatilityModeler(sample_returns)
        assert modeler.returns is not None
        assert modeler.window_size == 20

    def test_build_features_shape(self, sample_returns):
        """测试特征矩阵构建"""
        modeler = MLVolatilityModeler(sample_returns)
        X, y, names = modeler._build_features()
        assert X.shape[0] > 0
        assert X.shape[1] == 9  # 9个特征
        assert len(y) == X.shape[0]
        assert len(names) == 9

    def test_build_features_short_data_raises(self):
        """测试数据不足时特征构建抛出异常"""
        short_data = pd.Series(np.random.randn(15))
        modeler = MLVolatilityModeler(short_data, window_size=20)
        with pytest.raises(ValueError, match="数据长度"):
            modeler._build_features()

    def test_fit_random_forest(self, sample_returns):
        """测试Random Forest拟合"""
        modeler = MLVolatilityModeler(sample_returns)
        pred = modeler.fit_random_forest()
        assert 'RF' in modeler.models
        assert 'RF' in modeler.ic_scores
        assert 'RF' in modeler.predictions
        assert modeler.ic_scores['RF']['AIC'] < np.inf
        assert isinstance(pred, pd.Series)

    def test_fit_random_forest_ic_scores(self, sample_returns):
        """测试Random Forest IC分数字段"""
        modeler = MLVolatilityModeler(sample_returns)
        modeler.fit_random_forest()
        scores = modeler.ic_scores['RF']
        assert 'AIC' in scores
        assert 'BIC' in scores
        assert 'RMSE' in scores
        assert 'MAE' in scores
        assert 'converged' in scores
        assert scores['RMSE'] > 0
        assert scores['MAE'] > 0

    def test_fit_xgboost_installed(self, sample_returns):
        """测试XGBoost拟合（如果已安装）"""
        try:
            import xgboost  # noqa: F401
            modeler = MLVolatilityModeler(sample_returns)
            pred = modeler.fit_xgboost()
            assert 'XGBoost' in modeler.ic_scores
            assert modeler.ic_scores['XGBoost']['AIC'] < np.inf
        except ImportError:
            pytest.skip("XGBoost未安装")

    def test_fit_xgboost_not_installed(self, sample_returns):
        """测试XGBoost未安装时优雅降级"""
        # 如果XGBoost已安装则跳过
        try:
            import xgboost  # noqa: F401
            pytest.skip("XGBoost已安装，此测试针对未安装场景")
        except ImportError:
            modeler = MLVolatilityModeler(sample_returns)
            result = modeler.fit_xgboost()
            assert result is None
            assert modeler.ic_scores['XGBoost']['AIC'] == np.inf

    def test_fit_lstm_installed(self, sample_returns):
        """测试LSTM拟合（如果已安装）"""
        try:
            import tensorflow  # noqa: F401
            modeler = MLVolatilityModeler(sample_returns)
            pred = modeler.fit_lstm(epochs=5, batch_size=16)
            if pred is not None:
                assert 'LSTM' in modeler.ic_scores
                assert modeler.ic_scores['LSTM']['AIC'] < np.inf
        except ImportError:
            pytest.skip("TensorFlow未安装")

    def test_fit_lstm_not_installed(self, sample_returns):
        """测试TensorFlow未安装时优雅降级"""
        try:
            import tensorflow  # noqa: F401
            pytest.skip("TensorFlow已安装，此测试针对未安装场景")
        except ImportError:
            modeler = MLVolatilityModeler(sample_returns)
            result = modeler.fit_lstm()
            assert result is None
            assert modeler.ic_scores['LSTM']['AIC'] == np.inf

    def test_run_ml_tournament(self, sample_returns):
        """测试ML锦标赛"""
        modeler = MLVolatilityModeler(sample_returns)
        comparison, winner = modeler.run_ml_tournament()
        assert isinstance(comparison, dict)
        assert 'RF' in comparison
        # winner可能是RF、XGBoost或LSTM（取决于安装情况）
        if winner is not None:
            assert winner in comparison

    def test_compare_with_garch(self, sample_returns):
        """测试与GARCH对比"""
        modeler = MLVolatilityModeler(sample_returns)
        modeler.fit_random_forest()

        # 模拟GARCH IC分数
        garch_scores = {
            'GARCH': {'AIC': -500, 'BIC': -480, 'converged': True},
            'EGARCH': {'AIC': -520, 'BIC': -500, 'converged': True}
        }

        df, overall_winner = modeler.compare_with_garch(garch_scores)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 3  # 至少GARCH + EGARCH + RF

    def test_skewness_calculation(self, sample_returns):
        """测试偏度计算"""
        modeler = MLVolatilityModeler(sample_returns)
        x = np.array([1, 2, 3, 4, 5])
        skew = modeler._skewness(x)
        assert isinstance(skew, float)

    def test_kurtosis_calculation(self, sample_returns):
        """测试峰度计算"""
        modeler = MLVolatilityModeler(sample_returns)
        x = np.array([1, 2, 3, 4, 5])
        kurt = modeler._kurtosis(x)
        assert isinstance(kurt, float)

    def test_predictions_positive(self, sample_returns):
        """测试预测波动率为正"""
        modeler = MLVolatilityModeler(sample_returns)
        pred = modeler.fit_random_forest()
        assert (pred >= 0).all()  # 波动率必须非负

    def test_feature_importance(self, sample_returns):
        """测试特征重要性提取"""
        modeler = MLVolatilityModeler(sample_returns)
        modeler.fit_random_forest()
        model = modeler.models['RF']
        top_features = modeler._get_top_features(model, modeler.feature_names)
        assert len(top_features) == 3
        for name, importance in top_features:
            assert isinstance(name, str)
            assert importance > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
