"""
共享 pytest fixtures 和测试基础设施
v3.0 - CNLocalGovSpread 计量经济学框架

此文件提供:
1. sys.path 配置 - 所有测试文件不再需要单独配置
2. 共享 mock 数据 fixtures - 减少各测试文件的数据生成重复
3. Factory fixtures - 可配置参数的数据生成器
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import MagicMock

# 全局 sys.path 配置 - 所有测试文件无需单独插入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ============================================================================
# 基础配置 fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """标准 MOCK 数据源配置"""
    return {
        'SOURCE': 'MOCK',
        'TICKER': 'TEST',
        'START_DATE': '2020-01-01',
        'END_DATE': '2024-01-01',
        'MAD_THRESHOLD': 5.0,
        'SPREAD_COLUMN': 'spread_all',
        'GARCH_P': 1,
        'GARCH_Q': 1,
        'VaR_CONFIDENCE': 0.99,
        'EVT_THRESHOLD_PERCENTILE': 0.95
    }


@pytest.fixture
def mock_config_short():
    """短日期范围的 MOCK 配置（用于快速测试）"""
    config = {
        'SOURCE': 'MOCK',
        'START_DATE': '2022-01-01',
        'END_DATE': '2022-12-31',
        'MAD_THRESHOLD': 5.0,
        'SPREAD_COLUMN': 'spread_all'
    }
    return config


# ============================================================================
# Factory fixtures - 可配置参数的数据生成器
# ============================================================================

@pytest.fixture
def make_spread_data():
    """Factory fixture: 生成利差 DataFrame

    参数:
        n: 数据点数 (默认 300)
        base: 利差基准值 (默认 80)
        scale: 随机波动幅度 (默认 0.3)
        freq: 日期频率 (默认 'B' 工作日)
        start: 起始日期 (默认 '2020-01-01')
        seed: 随机种子 (默认 42)
        use_cumsum: 是否使用累积和 (默认 True)
    """
    def _make(n=300, base=80, scale=0.3, freq='B', start='2020-01-01',
              seed=42, use_cumsum=True):
        np.random.seed(seed)
        dates = pd.date_range(start, periods=n, freq=freq)
        if use_cumsum:
            spread = base + np.cumsum(np.random.randn(n) * scale)
        else:
            spread = base + np.random.randn(n) * scale
        return pd.DataFrame({'spread': spread}, index=dates)
    return _make


@pytest.fixture
def make_returns():
    """Factory fixture: 生成收益率 Series

    参数:
        n: 数据点数 (默认 200)
        scale: 波动幅度 (默认 0.02)
        seed: 随机种子 (默认 42)
        start: 起始日期 (默认 None, 无日期索引)
        freq: 日期频率 (默认 'B')
        distribution: 分布类型 'normal'/'fat_tail' (默认 'normal')
    """
    def _make(n=200, scale=0.02, seed=42, start=None, freq='B',
              distribution='normal'):
        np.random.seed(seed)
        if distribution == 'fat_tail':
            returns = np.random.standard_t(5, n) * scale
        else:
            returns = np.random.randn(n) * scale
        if start:
            dates = pd.date_range(start, periods=n, freq=freq)
            return pd.Series(returns, index=dates, name='returns')
        return pd.Series(returns, name='returns')
    return _make


@pytest.fixture
def make_dates():
    """Factory fixture: 生成日期范围

    参数:
        n: 天数 (默认 200)
        start: 起始日期 (默认 '2020-01-01')
        freq: 频率 (默认 'D')
    """
    def _make(n=200, start='2020-01-01', freq='D'):
        return pd.date_range(start, periods=n, freq=freq)
    return _make


@pytest.fixture
def make_volatility():
    """Factory fixture: 生成波动率 Series

    参数:
        n: 数据点数 (默认 200)
        scale: 基础波动率 (默认 0.02)
        seed: 随机种子 (默认 42)
        inject_high: 是否注入高波动率尾部 (默认 False)
        start: 起始日期 (默认 None)
        freq: 频率 (默认 'D')
    """
    def _make(n=200, scale=0.02, seed=42, inject_high=False,
              start=None, freq='D'):
        np.random.seed(seed)
        vol = np.abs(np.random.randn(n)) * scale + 0.01
        if inject_high:
            vol[-20:] = vol[-20:] * 3  # 注入高波动率尾部
        if start:
            dates = pd.date_range(start, periods=n, freq=freq)
            return pd.Series(vol, index=dates, name='volatility')
        return pd.Series(vol, name='volatility')
    return _make


@pytest.fixture
def make_smoothed_spread():
    """Factory fixture: 生成平滑利差 Series

    参数:
        n: 数据点数 (默认 200)
        base: 基准值 (默认 80)
        scale: 波动幅度 (默认 0.3)
        seed: 随机种子 (默认 42)
        start: 起始日期 (默认 None)
        freq: 频率 (默认 'D')
        use_cumsum: 是否使用累积和 (默认 True)
    """
    def _make(n=200, base=80, scale=0.3, seed=42, start=None, freq='D',
              use_cumsum=True):
        np.random.seed(seed)
        if use_cumsum:
            data = base + np.cumsum(np.random.randn(n) * scale)
        else:
            data = base + np.random.randn(n) * scale
        if start:
            dates = pd.date_range(start, periods=n, freq=freq)
            return pd.Series(data, index=dates, name='smoothed')
        return pd.Series(data, name='smoothed')
    return _make


@pytest.fixture
def make_signal_deviation():
    """Factory fixture: 生成信号偏离度 Series

    参数:
        n: 数据点数 (默认 200)
        seed: 随机种子 (默认 42)
        inject_extremes: 是否注入极端值 (默认 False)
        start: 起始日期 (默认 None)
        freq: 频率 (默认 'D')
    """
    def _make(n=200, seed=42, inject_extremes=False, start=None, freq='D'):
        np.random.seed(seed)
        deviation = np.random.randn(n)
        if inject_extremes:
            deviation[10] = 5.0   # 正向极端
            deviation[50] = -4.0  # 负向极端
        if start:
            dates = pd.date_range(start, periods=n, freq=freq)
            return pd.Series(deviation, index=dates, name='deviation')
        return pd.Series(deviation, name='deviation')
    return _make


# ============================================================================
# GARCH 模拟数据 fixtures
# ============================================================================

@pytest.fixture
def garch_returns():
    """GARCH(1,1) 模拟收益率数据 - 模拟真实金融序列特征

    生成过程: vol[t] = sqrt(0.1 + 0.15*r[t-1]^2 + 0.8*vol[t-1]^2)
    分布: t(5) 肥尾分布
    """
    np.random.seed(42)
    n = 500
    returns = np.zeros(n)
    vol = np.zeros(n)
    vol[0] = 0.5
    for t in range(1, n):
        vol[t] = np.sqrt(0.1 + 0.15 * returns[t-1]**2 + 0.8 * vol[t-1]**2)
        returns[t] = np.random.standard_t(5) * vol[t]
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    return pd.Series(returns, index=dates)


@pytest.fixture
def fat_tail_returns():
    """标准 t(5) 肥尾分布收益率 - 用于 EVT 和场景分析测试"""
    np.random.seed(42)
    return pd.Series(np.random.standard_t(5, 500) * 0.5)


@pytest.fixture
def spread_series():
    """利差累积和序列 - 用于卡尔曼滤波测试"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='B')
    spread = 100 + np.cumsum(np.random.randn(500) * 0.1)
    return pd.Series(spread, index=dates)


# ============================================================================
# 边界测试数据 fixtures
# ============================================================================

@pytest.fixture
def short_returns():
    """极短收益率数据 (n=10) - 用于边界条件测试"""
    np.random.seed(42)
    return pd.Series(np.random.randn(10) * 0.1)


@pytest.fixture
def minimal_returns():
    """最小长度收益率数据 (n=30) - 用于 FIGARCH 边界测试"""
    np.random.seed(99)
    return pd.Series(np.random.randn(30) * 0.1)


@pytest.fixture
def very_short_returns():
    """极短收益率数据 (n=20) - 用于 FIGARCH 边界测试"""
    np.random.seed(77)
    return pd.Series(np.random.randn(20) * 0.1)


# ============================================================================
# Mock 对象 fixtures - 用于模块集成测试
# ============================================================================

@pytest.fixture
def mock_kalman():
    """模拟 KalmanSignalExtractor 对象"""
    mock = MagicMock()
    mock.smoothed_state = pd.Series(np.random.randn(100) * 0.5 + 80)
    mock.get_signal_deviation.return_value = pd.Series(np.random.randn(100))
    return mock


@pytest.fixture
def mock_vol_modeler():
    """模拟 VolatilityModeler 对象"""
    mock = MagicMock()
    mock.ic_scores = {
        'GARCH': {'AIC': -100, 'BIC': -95},
        'EGARCH': {'AIC': -102, 'BIC': -97},
        'GJR-GARCH': {'AIC': -101, 'BIC': -96}
    }
    mock.winner_model = 'EGARCH'
    mock.get_conditional_volatility.return_value = pd.Series(
        np.abs(np.random.randn(100)) * 0.02 + 0.01
    )
    return mock


@pytest.fixture
def mock_evt():
    """模拟 EVTRiskAnalyzer 对象"""
    mock = MagicMock()
    mock.var = -0.05
    mock.es = -0.08
    mock.threshold = 0.02
    mock.gpd_params = {'shape': 0.3, 'scale': 0.01}
    return mock


@pytest.fixture
def sample_alerts():
    """标准测试告警列表"""
    from datetime import datetime
    return [
        {
            'type': 'VaR', 'level': 'danger',
            'message': 'VaR breach', 'timestamp': datetime.now()
        },
        {
            'type': 'spread', 'level': 'warning',
            'message': 'Spread high', 'timestamp': datetime.now()
        },
        {
            'type': 'trend', 'level': 'success',
            'message': 'Trend normal', 'timestamp': datetime.now()
        },
        {
            'type': 'volatility', 'level': 'info',
            'message': 'Vol low', 'timestamp': datetime.now()
        }
    ]


# ============================================================================
# 长记忆序列 fixture
# ============================================================================

@pytest.fixture
def long_memory_returns():
    """ARFIMA(0,0.4,0) 长记忆收益率 - 用于 FIGARCH 测试"""
    np.random.seed(123)
    n = 500
    d = 0.4
    noise = np.random.randn(n + 100)[:n]

    # 分数差分滤波器系数
    pi_coeffs = [1.0]
    for k in range(1, 100):
        pi_coeffs.append(pi_coeffs[-1] * (k - 1 - d) / k)

    # 应用分数差分
    series = np.zeros(n)
    for t in range(n):
        for k in range(min(t + 1, len(pi_coeffs))):
            series[t] += pi_coeffs[k] * noise[t - k]

    return pd.Series(series * 0.1)


# ============================================================================
# 注册 marker
# ============================================================================

def pytest_configure(config):
    """注册自定义 pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m not slow')"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "dashboard: marks dashboard-related tests"
    )
    config.addinivalue_line(
        "markers", "requires_xgboost: requires xgboost package"
    )
    config.addinivalue_line(
        "markers", "requires_tensorflow: requires tensorflow package"
    )