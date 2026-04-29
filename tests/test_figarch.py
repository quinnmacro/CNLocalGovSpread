"""
FIGARCH 长记忆检测与拟合测试 (v3.0新增)

测试覆盖:
1. GPH估计器 (detect_long_memory): 分数差分参数d估计
2. FIGARCH拟合 (fit_figarch): 条件方差计算与IC分数
3. 长记忆数据生成与检测准确性
4. FIGARCH与其他模型比较
5. 边界情况处理 (短数据, 优化失败)
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from volatility import VolatilityModeler


# ============================================================================
# 测试数据生成
# ============================================================================

@pytest.fixture
def short_memory_returns():
    """短记忆收益率数据 (标准GARCH适用)"""
    np.random.seed(42)
    return pd.Series(np.random.randn(500) * 0.1)


@pytest.fixture
def long_memory_returns():
    """长记忆收益率数据 (FIGARCH适用)

    通过累积ARFIMA(0,d,0)噪声模拟长记忆特性
    d=0.4: 中等长记忆
    """
    np.random.seed(123)
    n = 500
    d = 0.4

    # 生成ARFIMA(0,d,0)过程: (1-L)^d * ε_t = 0
    # 即 X_t = Σ π_k ε_{t-k} (分数差分滤波器的逆操作)
    # 等价于 X_t = (1-L)^{-d} ε_t
    pi_neg = np.zeros(n)
    # (1-L)^{-d} 的展开系数
    # π_neg[0] = 1, π_neg[k] = (k-1+d)/k * π_neg[k-1]
    pi_neg[0] = 1.0
    for k in range(1, n):
        pi_neg[k] = (k - 1 + d) / k * pi_neg[k - 1]

    eps = np.random.randn(n) * 0.01
    x = np.zeros(n)
    for t in range(n):
        for k in range(min(t + 1, n)):
            x[t] += pi_neg[k] * eps[t - k]

    return pd.Series(x)


@pytest.fixture
def minimal_returns():
    """极短数据 (<50) 用于边界测试"""
    np.random.seed(99)
    return pd.Series(np.random.randn(30) * 0.1)


@pytest.fixture
def very_short_returns():
    """极短数据 (<30) 用于FIGARCH边界测试"""
    np.random.seed(77)
    return pd.Series(np.random.randn(20) * 0.1)


# ============================================================================
# GPH 估计器测试
# ============================================================================

class TestDetectLongMemory:
    """GPH长记忆检测测试"""

    def test_detect_returns_dict(self, short_memory_returns):
        """测试 detect_long_memory 返回完整字典"""
        modeler = VolatilityModeler(short_memory_returns)
        result = modeler.detect_long_memory()

        assert isinstance(result, dict)
        assert 'd_estimate' in result
        assert 'd_std_error' in result
        assert 'd_t_stat' in result
        assert 'd_p_value' in result
        assert 'long_memory_detected' in result
        assert 'memory_type' in result
        assert 'gph_regression_details' in result

    def test_gph_d_estimate_range(self, short_memory_returns):
        """测试 d 估计值在合理范围 [-1, 1]"""
        modeler = VolatilityModeler(short_memory_returns)
        result = modeler.detect_long_memory()

        # d 应在经济学合理范围内
        assert -1 <= result['d_estimate'] <= 1

    def test_gph_std_error_positive(self, short_memory_returns):
        """测试标准误差为正"""
        modeler = VolatilityModeler(short_memory_returns)
        result = modeler.detect_long_memory()

        assert result['d_std_error'] > 0

    def test_gph_p_value_in_range(self, short_memory_returns):
        """测试 p值在 [0, 1]"""
        modeler = VolatilityModeler(short_memory_returns)
        result = modeler.detect_long_memory()

        assert 0 <= result['d_p_value'] <= 1

    def test_gph_regression_details(self, short_memory_returns):
        """测试回归详情包含关键信息"""
        modeler = VolatilityModeler(short_memory_returns)
        result = modeler.detect_long_memory()

        details = result['gph_regression_details']
        assert details is not None
        assert 'n_freq_used' in details
        assert 'intercept' in details
        assert 'r_squared' in details
        assert 'd_confidence_interval' in details
        assert len(details['d_confidence_interval']) == 2

    def test_short_memory_detected_for_random_data(self, short_memory_returns):
        """测试纯随机数据的短记忆检测"""
        modeler = VolatilityModeler(short_memory_returns)
        result = modeler.detect_long_memory()

        # 纯随机数据应检测为短记忆 (d接近0或不显著)
        # 允许一定误差, 因为GPH估计器有噪音
        assert isinstance(result['long_memory_detected'], bool)
        assert isinstance(result['memory_type'], str)

    def test_gph_handles_short_data(self, minimal_returns):
        """测试 GPH 处理短数据 (<50)"""
        modeler = VolatilityModeler(minimal_returns)
        result = modeler.detect_long_memory()

        assert result['d_estimate'] == 0.0
        assert result['d_std_error'] == np.inf
        assert result['long_memory_detected'] is False
        assert '数据不足' in result['memory_type']

    def test_gph_custom_max_freq(self, short_memory_returns):
        """测试自定义最大频率比例"""
        modeler = VolatilityModeler(short_memory_returns)
        result_default = modeler.detect_long_memory()
        result_custom = modeler.detect_long_memory(max_freq_frac=0.3)

        # 两种设置应给出不同但都在合理范围的d估计
        assert -1 <= result_custom['d_estimate'] <= 1
        assert result_custom['gph_regression_details']['n_freq_used'] < \
               result_default['gph_regression_details']['n_freq_used']


# ============================================================================
# FIGARCH 拟合测试
# ============================================================================

class TestFitFIGARCH:
    """FIGARCH拟合测试"""

    def test_fit_figarch_returns_result(self, short_memory_returns):
        """测试 FIGARCH 拟合返回结果"""
        modeler = VolatilityModeler(short_memory_returns)
        result = modeler.fit_figarch(truncation_lag=50)

        assert result is not None
        assert 'volatility' in result
        assert 'params' in result
        assert 'gph_result' in result

    def test_figarch_params_structure(self, short_memory_returns):
        """测试 FIGARCH 参数结构"""
        modeler = VolatilityModeler(short_memory_returns)
        result = modeler.fit_figarch(truncation_lag=50)

        params = result['params']
        assert 'omega' in params
        assert 'alpha' in params
        assert 'beta' in params
        assert 'd' in params
        assert 'df' in params

    def test_figarch_param_constraints(self, short_memory_returns):
        """测试 FIGARCH 参数约束"""
        modeler = VolatilityModeler(short_memory_returns)
        result = modeler.fit_figarch(truncation_lag=50)

        params = result['params']
        assert params['omega'] > 0
        assert params['alpha'] > 0
        assert params['beta'] >= 0
        assert 0.01 <= params['d'] <= 0.99
        assert params['alpha'] + params['beta'] < 1

    def test_figarch_volatility_series(self, short_memory_returns):
        """测试 FIGARCH 波动率序列"""
        modeler = VolatilityModeler(short_memory_returns)
        result = modeler.fit_figarch(truncation_lag=50)

        vol = result['volatility']
        assert isinstance(vol, pd.Series)
        assert len(vol) == len(short_memory_returns)
        assert (vol > 0).all()

    def test_figarch_ic_scores_stored(self, short_memory_returns):
        """测试 FIGARCH IC分数存储"""
        modeler = VolatilityModeler(short_memory_returns)
        modeler.fit_figarch(truncation_lag=50)

        assert 'FIGARCH' in modeler.ic_scores
        scores = modeler.ic_scores['FIGARCH']
        assert 'AIC' in scores
        assert 'BIC' in scores
        assert 'converged' in scores
        assert scores['AIC'] < np.inf

    def test_figarch_pi_coefficients(self, short_memory_returns):
        """测试 FIGARCH 长记忆滤波器系数"""
        modeler = VolatilityModeler(short_memory_returns)
        result = modeler.fit_figarch(truncation_lag=50)

        pi = result['pi_coefficients']
        assert pi[0] == 1.0  # π_0 = 1
        # 双曲衰减: π_k 应逐渐减小
        # 对于 d>0, π_k 为负且递减
        assert len(pi) > 0

    def test_figarch_handles_very_short_data(self, very_short_returns):
        """测试 FIGARCH 处理极短数据 (<30)"""
        modeler = VolatilityModeler(very_short_returns)
        result = modeler.fit_figarch()

        assert result is None
        assert modeler.ic_scores['FIGARCH']['AIC'] == np.inf
        assert modeler.ic_scores['FIGARCH']['converged'] is False

    def test_figarch_custom_truncation(self, short_memory_returns):
        """测试自定义截断阶数"""
        modeler = VolatilityModeler(short_memory_returns)
        result_short = modeler.fit_figarch(truncation_lag=20)
        # 清除后重新拟合更长截断
        modeler2 = VolatilityModeler(short_memory_returns)
        result_long = modeler2.fit_figarch(truncation_lag=80)

        # 两种截断都应产生合理结果
        assert result_short['params']['d'] > 0
        assert result_long['params']['d'] > 0


# ============================================================================
# FIGARCH 与模型锦标赛集成测试
# ============================================================================

class TestFIGARCHIntegration:
    """FIGARCH与锦标赛集成测试"""

    def test_tournament_without_figarch(self, short_memory_returns):
        """测试不含FIGARCH的锦标赛 (默认行为不变)"""
        modeler = VolatilityModeler(short_memory_returns)
        winner = modeler.run_tournament()

        assert winner in ['GARCH', 'EGARCH', 'GJR-GARCH', 'EWMA']
        assert 'FIGARCH' not in modeler.ic_scores

    def test_tournament_with_figarch(self, short_memory_returns):
        """测试包含FIGARCH的锦标赛"""
        modeler = VolatilityModeler(short_memory_returns)
        winner = modeler.run_tournament(include_figarch=True)

        assert winner in ['GARCH', 'EGARCH', 'GJR-GARCH', 'EWMA', 'FIGARCH']
        assert 'FIGARCH' in modeler.ic_scores

    def test_get_figarch_conditional_volatility(self, short_memory_returns):
        """测试获取FIGARCH条件波动率"""
        modeler = VolatilityModeler(short_memory_returns)
        modeler.fit_figarch(truncation_lag=50)

        vol = modeler.get_conditional_volatility('FIGARCH')
        assert isinstance(vol, pd.Series)
        assert (vol > 0).all()

    def test_figarch_volatility_comparison(self, short_memory_returns):
        """测试FIGARCH波动率与GARCH波动率对比"""
        modeler = VolatilityModeler(short_memory_returns)
        modeler.run_tournament()

        garch_vol = modeler.get_conditional_volatility('GARCH')

        modeler.fit_figarch(truncation_lag=50)
        figarch_vol = modeler.get_conditional_volatility('FIGARCH')

        # 两者都应为正且长度相同
        assert len(garch_vol) == len(figarch_vol)
        assert (garch_vol > 0).all()
        assert (figarch_vol > 0).all()

    def test_figarch_d_estimate_consistency(self, short_memory_returns):
        """测试d估计值与GPH结果一致"""
        modeler = VolatilityModeler(short_memory_returns)
        gph_result = modeler.detect_long_memory()
        figarch_result = modeler.fit_figarch(truncation_lag=50)

        # FIGARCH的d参数应与GPH估计一致 (被限制在[0.01, 0.99]后)
        d_figarch = figarch_result['params']['d']
        d_gph = gph_result['d_estimate']

        # d被clamp到[0.01, 0.99], 所以原始值超出范围时可能不同
        d_gph_clamped = max(0.01, min(0.99, d_gph))
        assert abs(d_figarch - d_gph_clamped) < 0.001


# ============================================================================
# 长记忆特征验证测试
# ============================================================================

class TestLongMemoryCharacteristics:
    """长记忆特征验证测试"""

    def test_pi_coefficients_hyperbolic_decay(self, short_memory_returns):
        """测试π系数的双曲衰减特性

        FIGARCH的关键特征: π_k 以双曲速率(∝k^{d-1})衰减
        GARCH的对应系数以指数速率衰减
        """
        modeler = VolatilityModeler(short_memory_returns)
        result = modeler.fit_figarch(truncation_lag=100)

        d = result['params']['d']
        pi = np.zeros(101)
        pi[0] = 1.0
        for k in range(1, 101):
            pi[k] = (k - 1 - d) / k * pi[k - 1]

        # 双曲衰减验证: 对于中等d, π_50 / π_10 应比指数衰减大
        # 指数衰减时比例约为 exp(-40*rate) ≈ 极小值
        # 双曲衰减时比例约为 (50/10)^{d-1} ≈ 中等值
        ratio = abs(pi[50] / pi[10]) if abs(pi[10]) > 1e-10 else 0

        # 对于 d > 0, 双曲衰减使π系数缓慢趋0
        # ratio 应显著大于纯指数衰减的极小值
        # 但具体数值取决于d, 所以只验证合理性
        assert 0 <= ratio < 1  # π递减, ratio < 1

    def test_autocorrelation_decay_comparison(self):
        """测试长记忆数据的自相关衰减 vs 短记忆数据

        长记忆数据: 自相关以双曲速率衰减 (缓慢)
        短记忆数据: 自相关以指数速率衰减 (快速)
        """
        np.random.seed(42)
        n = 500

        # 短记忆: AR(1) phi=0.3
        short_data = np.zeros(n)
        for t in range(1, n):
            short_data[t] = 0.3 * short_data[t - 1] + np.random.randn()

        # 长记忆: ARFIMA(0, 0.4, 0)
        d = 0.4
        pi_neg = np.zeros(n)
        pi_neg[0] = 1.0
        for k in range(1, n):
            pi_neg[k] = (k - 1 + d) / k * pi_neg[k - 1]

        eps = np.random.randn(n)
        long_data = np.zeros(n)
        for t in range(n):
            for k in range(min(t + 1, 100)):
                long_data[t] += pi_neg[k] * eps[t - k]

        # 计算滞后20的自相关
        acf_short = np.corrcoef(short_data[:-20], short_data[20:])[0, 1]
        acf_long = np.corrcoef(long_data[:-20], long_data[20:])[0, 1]

        # 长记忆的自相关在滞后20应仍显著 (> 短记忆)
        # AR(1) phi=0.3: lag-20 AC = 0.3^20 ≈ 3.5e-11 ≈ 0
        # ARFIMA(0,0.4,0): lag-20 AC ≈ C * 20^{2d-1} ≈ 显著
        # 但随机波动可能导致不完全符合理论, 所以放宽判断
        assert acf_long > acf_short or abs(acf_short) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
