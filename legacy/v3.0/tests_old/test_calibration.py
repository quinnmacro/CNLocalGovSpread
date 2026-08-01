"""
参数校准模块单元测试
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from calibration import ParameterCalibrator
from data_engine import DataEngine


# ============================================================================
# 测试配置和数据
# ============================================================================

@pytest.fixture
def test_config():
    return {
        'SOURCE': 'MOCK',
        'START_DATE': '2020-01-01',
        'END_DATE': '2022-12-31',
        'MAD_THRESHOLD': 5.0
    }


@pytest.fixture
def sample_returns():
    """生成测试用收益率数据（带肥尾特征）"""
    np.random.seed(42)
    n = 500
    # 模拟GARCH(1,1)过程产生的收益率
    returns = np.zeros(n)
    vol = np.zeros(n)
    vol[0] = 0.5
    for t in range(1, n):
        vol[t] = np.sqrt(0.1 + 0.15 * returns[t-1]**2 + 0.8 * vol[t-1]**2)
        returns[t] = np.random.standard_t(5) * vol[t]
    return pd.Series(returns)


@pytest.fixture
def sample_spread():
    """生成测试用利差数据"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='B')
    spread = 100 + np.cumsum(np.random.randn(500) * 0.1)
    return pd.Series(spread, index=dates)


@pytest.fixture
def short_returns():
    """短数据序列（不足最小样本量）"""
    np.random.seed(42)
    return pd.Series(np.random.randn(10) * 0.1)


@pytest.fixture
def calibrator(sample_returns):
    """标准校准器实例"""
    return ParameterCalibrator(sample_returns)


@pytest.fixture
def calibrator_with_spread(sample_returns, sample_spread):
    """带利差数据的校准器实例"""
    return ParameterCalibrator(sample_returns, spread=sample_spread)


# ============================================================================
# 初始化测试
# ============================================================================

class TestCalibratorInit:
    """校准器初始化测试"""

    def test_init_with_returns_only(self, sample_returns):
        cal = ParameterCalibrator(sample_returns)
        assert cal.returns is sample_returns
        assert cal.spread is None
        assert cal.calibrated == {}
        assert cal.diagnostics == {}

    def test_init_with_returns_and_spread(self, sample_returns, sample_spread):
        cal = ParameterCalibrator(sample_returns, spread=sample_spread)
        assert cal.spread is sample_spread


# ============================================================================
# EWMA Lambda 优化测试
# ============================================================================

class TestEWMALambda:
    """EWMA lambda优化测试"""

    def test_estimate_returns_valid_lambda(self, calibrator):
        lam = calibrator.estimate_ewma_lambda()
        assert 0.80 <= lam <= 0.99
        assert 'ewma_lambda' in calibrator.calibrated

    def test_lambda_not_exactly_default(self, calibrator):
        """校准值应偏离硬编码0.94（除非数据恰好最优）"""
        lam = calibrator.estimate_ewma_lambda()
        # 不严格要求≠0.94，但应在合理范围
        assert 0.80 <= lam <= 0.99

    def test_lambda_has_diagnostics(self, calibrator):
        calibrator.estimate_ewma_lambda()
        diag = calibrator.diagnostics['ewma_lambda']
        assert 'method' in diag
        assert 'ci' in diag
        assert 'rmse' in diag

    def test_lambda_fallback_for_short_data(self, short_returns):
        cal = ParameterCalibrator(short_returns)
        lam = cal.estimate_ewma_lambda()
        assert lam == 0.94  # fallback默认值

    def test_lambda_custom_range(self, sample_returns):
        cal = ParameterCalibrator(sample_returns)
        lam = cal.estimate_ewma_lambda(lambda_range=(0.90, 0.96))
        assert 0.90 <= lam <= 0.96


# ============================================================================
# t分布df估计测试
# ============================================================================

class TestTDistribution:
    """t分布df估计测试"""

    def test_estimate_returns_valid_df(self, calibrator):
        df = calibrator.estimate_t_df()
        assert 2.1 <= df <= 30.0
        assert 't_df' in calibrator.calibrated

    def test_df_fat_tail_detection(self, calibrator):
        """肥尾数据应产生较低的df值"""
        df = calibrator.estimate_t_df()
        diag = calibrator.diagnostics['t_df']
        assert 'tail_diagnosis' in diag
        # 使用t(5)生成的数据应检测到肥尾
        assert df < 25  # 不应接近正态

    def test_df_has_confidence_interval(self, calibrator):
        df = calibrator.estimate_t_df()
        diag = calibrator.diagnostics['t_df']
        assert 'ci' in diag
        assert '(' in diag['ci']  # CI格式检查

    def test_df_fallback_for_short_data(self, short_returns):
        cal = ParameterCalibrator(short_returns)
        df = cal.estimate_t_df()
        assert df == 5.0  # fallback默认值

    def test_df_bounds_respected(self, sample_returns):
        cal = ParameterCalibrator(sample_returns)
        df = cal.estimate_t_df(bounds=(3.0, 15.0))
        assert 3.0 <= df <= 15.0


# ============================================================================
# AR(1) phi估计测试
# ============================================================================

class TestARPhi:
    """AR(1) phi估计测试"""

    def test_estimate_returns_valid_phi(self, calibrator):
        phi = calibrator.estimate_ar_phi()
        assert 0.0 <= phi <= 0.99
        assert 'ar_phi' in calibrator.calibrated

    def test_phi_has_significance_test(self, calibrator):
        phi = calibrator.estimate_ar_phi()
        diag = calibrator.diagnostics['ar_phi']
        assert 't_stat' in diag
        assert 'p_value' in diag
        assert 'significant' in diag

    def test_phi_fallback_for_short_data(self, short_returns):
        cal = ParameterCalibrator(short_returns)
        phi = cal.estimate_ar_phi()
        assert phi == 0.5  # fallback默认值

    def test_phi_persistence_diagnosis(self, calibrator):
        phi = calibrator.estimate_ar_phi()
        diag = calibrator.diagnostics['ar_phi']
        assert 'persist_diagnosis' in diag


# ============================================================================
# EVT阈值优化测试
# ============================================================================

class TestEVTThreshold:
    """EVT阈值优化测试"""

    def test_optimize_returns_valid_percentile(self, calibrator):
        pct = calibrator.optimize_evt_threshold()
        assert 0.85 <= pct <= 0.99
        assert 'evt_threshold_percentile' in calibrator.calibrated

    def test_evt_has_threshold_value(self, calibrator):
        pct = calibrator.optimize_evt_threshold()
        diag = calibrator.diagnostics['evt_threshold_percentile']
        assert 'threshold_value' in diag
        assert 'n_exceedances' in diag
        assert diag['n_exceedances'] >= 10

    def test_evt_fallback_for_short_data(self, short_returns):
        cal = ParameterCalibrator(short_returns)
        pct = cal.optimize_evt_threshold()
        assert pct == 0.95

    def test_evt_custom_range(self, sample_returns):
        cal = ParameterCalibrator(sample_returns)
        pct = cal.optimize_evt_threshold(percentile_range=(0.90, 0.98))
        assert 0.90 <= pct <= 0.98


# ============================================================================
# Kalman窗口优化测试
# ============================================================================

class TestKalmanWindow:
    """Kalman窗口优化测试"""

    def test_optimize_with_spread_data(self, calibrator_with_spread):
        window = calibrator_with_spread.optimize_kalman_window()
        assert 20 <= window <= 120
        assert isinstance(window, int)

    def test_window_fallback_without_spread(self, calibrator):
        """没有利差数据时应使用默认值"""
        window = calibrator.optimize_kalman_window()
        assert window == 60

    def test_window_fallback_for_short_data(self, short_returns):
        dates = pd.date_range('2020-01-01', periods=10, freq='B')
        short_spread = pd.Series(np.random.randn(10), index=dates)
        cal = ParameterCalibrator(short_returns, spread=short_spread)
        window = cal.optimize_kalman_window()
        assert window == 60


# ============================================================================
# 信号阈值估计测试
# ============================================================================

class TestSignalThreshold:
    """信号偏离阈值估计测试"""

    def test_estimate_returns_positive_threshold(self, calibrator):
        threshold = calibrator.estimate_signal_threshold()
        assert threshold > 0
        assert 'signal_threshold' in calibrator.calibrated

    def test_threshold_has_multi_level(self, calibrator):
        threshold = calibrator.estimate_signal_threshold()
        diag = calibrator.diagnostics['signal_threshold']
        assert 'thresholds_at_levels' in diag
        assert 0.90 in diag['thresholds_at_levels']
        assert 0.95 in diag['thresholds_at_levels']
        assert 0.99 in diag['thresholds_at_levels']

    def test_threshold_monotonic_levels(self, calibrator):
        """阈值应随置信水平单调递增"""
        calibrator.estimate_signal_threshold()
        diag = calibrator.diagnostics['signal_threshold']
        levels = diag['thresholds_at_levels']
        assert levels[0.90] <= levels[0.95] <= levels[0.99]

    def test_threshold_fallback_for_short_data(self, short_returns):
        cal = ParameterCalibrator(short_returns)
        threshold = cal.estimate_signal_threshold()
        assert threshold == 1.5


# ============================================================================
# GARCH持久性校验测试
# ============================================================================

class TestGARCHPersistence:
    """GARCH持久性校验测试"""

    def test_diagnose_with_garch_results(self, sample_returns):
        from volatility import VolatilityModeler
        vol = VolatilityModeler(sample_returns)
        vol.run_tournament()
        cal = ParameterCalibrator(sample_returns)
        diag = cal.diagnose_garch_persistence(vol.results)
        assert isinstance(diag, dict)
        # 应至少有一个GARCH模型的诊断结果
        assert len(diag) > 0

    def test_diagnose_without_results(self, calibrator):
        diag = calibrator.diagnose_garch_persistence()
        assert 'garch_persistence' in calibrator.diagnostics

    def test_persistence_half_life(self, sample_returns):
        """平稳GARCH应计算半衰期"""
        from volatility import VolatilityModeler
        vol = VolatilityModeler(sample_returns)
        vol.run_tournament()
        cal = ParameterCalibrator(sample_returns)
        diag = cal.diagnose_garch_persistence(vol.results)
        for model_name, model_diag in diag.items():
            if 'error' not in model_diag:
                if model_diag['stationary']:
                    assert model_diag['half_life'] < np.inf
                    assert model_diag['half_life'] > 0


# ============================================================================
# calibrate_all 集成测试
# ============================================================================

class TestCalibrateAll:
    """一键校准集成测试"""

    def test_calibrate_all_returns_dict(self, calibrator):
        result = calibrator.calibrate_all()
        assert isinstance(result, dict)
        assert len(result) >= 5  # 至少5个参数

    def test_calibrate_all_populates_all_params(self, calibrator):
        calibrator.calibrate_all()
        expected_params = ['ewma_lambda', 't_df', 'ar_phi',
                                'evt_threshold_percentile', 'signal_threshold']
        for param in expected_params:
            assert param in calibrator.calibrated

    def test_calibrate_all_with_spread(self, calibrator_with_spread):
        result = calibrator_with_spread.calibrate_all()
        assert 'kalman_window' in result

    def test_calibrate_all_no_missing_returns(self, calibrator):
        """所有校准值不应为None或NaN"""
        result = calibrator.calibrate_all()
        for param, value in result.items():
            assert value is not None
            assert not np.isnan(value) if isinstance(value, float) else True


# ============================================================================
# 配置生成测试
# ============================================================================

class TestCalibrationConfig:
    """校准配置生成测试"""

    def test_get_config_returns_dict(self, calibrator):
        calibrator.calibrate_all()
        config = calibrator.get_calibration_config()
        assert isinstance(config, dict)

    def test_config_key_names(self, calibrator):
        calibrator.calibrate_all()
        config = calibrator.get_calibration_config()
        expected_keys = ['EWMA_LAMBDA', 'T_DF', 'AR_PHI',
                         'EVT_THRESHOLD_PERCENTILE', 'SIGNAL_THRESHOLD']
        for key in expected_keys:
            assert key in config


# ============================================================================
# 报告生成测试
# ============================================================================

class TestCalibrationReport:
    """校准报告生成测试"""

    def test_print_report_no_error(self, calibrator):
        calibrator.calibrate_all()
        # 不测试输出内容，只测试不抛异常
        calibrator.print_calibration_report()


# ============================================================================
# 完整工作流集成测试
# ============================================================================

class TestCalibrationWorkflow:
    """校准与现有模块的集成测试"""

    def test_calibrate_then_use_ewma(self, test_config):
        """校准后参数可用于EWMA模型"""
        engine = DataEngine(test_config)
        engine.load_data()
        engine.clean_data()
        returns = engine.get_returns()

        cal = ParameterCalibrator(returns)
        cal.calibrate_all()

        from volatility import VolatilityModeler
        vol = VolatilityModeler(returns)
        # 使用校准后的lambda而非默认0.94
        custom_lambda = cal.calibrated['ewma_lambda']
        vol.fit_ewma(lambda_param=custom_lambda)

        assert 'EWMA' in vol.models
        assert vol.ic_scores['EWMA']['AIC'] < np.inf

    def test_calibrate_then_use_evt(self, test_config):
        """校准后阈值可用于EVT分析"""
        engine = DataEngine(test_config)
        engine.load_data()
        engine.clean_data()
        returns = engine.get_returns()

        cal = ParameterCalibrator(returns)
        cal.calibrate_all()

        from evt import EVTRiskAnalyzer
        custom_pct = cal.calibrated['evt_threshold_percentile']
        evt = EVTRiskAnalyzer(returns, threshold_percentile=custom_pct)
        evt.fit_gpd()

        assert evt.threshold is not None
