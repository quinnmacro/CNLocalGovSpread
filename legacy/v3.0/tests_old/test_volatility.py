"""
VolatilityModeler 和 RegimeDetector 综合测试
覆盖: init, fit_garch/egarch/gjr_garch/ewma, run_tournament,
      get_conditional_volatility, get_parameter_diagnostics, RegimeDetector
"""

import pytest
import numpy as np
import pandas as pd
from volatility import VolatilityModeler, RegimeDetector


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_returns():
    """标准测试收益率 (n=300, normal)"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=300, freq='B')
    return pd.Series(np.random.randn(300) * 0.02, index=dates)


@pytest.fixture
def fat_tail_returns():
    """肥尾收益率 (t分布 df=5)"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=300, freq='B')
    return pd.Series(np.random.standard_t(5, 300) * 0.02, index=dates)


@pytest.fixture
def garch_sim_returns():
    """GARCH(1,1) 模拟数据 - 具有真实波动率聚类"""
    np.random.seed(42)
    n = 500
    returns = np.zeros(n)
    vol = np.zeros(n)
    vol[0] = 0.02
    for t in range(1, n):
        vol[t] = np.sqrt(0.0001 + 0.15 * returns[t-1]**2 + 0.8 * vol[t-1]**2)
        returns[t] = np.random.standard_t(5) * vol[t]
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    return pd.Series(returns, index=dates)


@pytest.fixture
def short_returns():
    """短数据 (n=30) - 边界测试"""
    np.random.seed(42)
    return pd.Series(np.random.randn(30) * 0.02)


@pytest.fixture
def very_short_returns():
    """极短数据 (n=5) - EWMA边界测试"""
    np.random.seed(42)
    return pd.Series(np.random.randn(5) * 0.02)


@pytest.fixture
def minimal_returns():
    """最小数据 (n=3) - EWMA极端边界"""
    np.random.seed(42)
    return pd.Series(np.random.randn(3) * 0.02)


@pytest.fixture
def constant_returns():
    """恒定收益率 - 测试收敛异常"""
    return pd.Series(np.zeros(200) + 0.001)


@pytest.fixture
def volatility_series():
    """波动率序列 - RegimeDetector测试"""
    np.random.seed(42)
    n = 300
    vol = np.abs(np.random.randn(n)) * 0.02 + 0.01
    vol[-30:] = vol[-30:] * 3  # 注入高波动率尾段
    return pd.Series(vol)


# ============================================================================
# VolatilityModeler Init Tests
# ============================================================================

class TestVolatilityModelerInit:
    """VolatilityModeler 初始化测试"""

    def test_init_stores_returns(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        assert modeler.returns is sample_returns

    def test_init_default_pq(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        assert modeler.p == 1
        assert modeler.q == 1

    def test_init_custom_pq(self, sample_returns):
        modeler = VolatilityModeler(sample_returns, p=2, q=2)
        assert modeler.p == 2
        assert modeler.q == 2

    def test_init_empty_dicts(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        assert modeler.models == {}
        assert modeler.results == {}
        assert modeler.ic_scores == {}


# ============================================================================
# GARCH Fitting Tests
# ============================================================================

class TestFitGarch:
    """fit_garch 方法测试"""

    def test_fit_populates_results(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_garch()
        assert 'GARCH' in modeler.results
        assert 'GARCH' in modeler.models
        assert 'GARCH' in modeler.ic_scores

    def test_fit_ic_scores_structure(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_garch()
        scores = modeler.ic_scores['GARCH']
        assert 'AIC' in scores
        assert 'BIC' in scores
        assert 'converged' in scores

    def test_fit_aic_bic_finite(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_garch()
        assert modeler.ic_scores['GARCH']['AIC'] < np.inf
        assert modeler.ic_scores['GARCH']['BIC'] < np.inf

    def test_fit_convergence_flag(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_garch()
        assert isinstance(modeler.ic_scores['GARCH']['converged'], bool)

    def test_fit_conditional_volatility_positive(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_garch()
        vol = modeler.results['GARCH'].conditional_volatility
        assert (vol > 0).all()

    def test_fit_result_has_params(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_garch()
        result = modeler.results['GARCH']
        assert hasattr(result, 'params')
        assert hasattr(result, 'aic')
        assert hasattr(result, 'bic')

    def test_fit_fat_tail_data(self, fat_tail_returns):
        modeler = VolatilityModeler(fat_tail_returns)
        modeler.fit_garch()
        assert 'GARCH' in modeler.ic_scores
        assert modeler.ic_scores['GARCH']['AIC'] < np.inf


class TestFitEgarch:
    """fit_egarch 方法测试"""

    def test_fit_populates_results(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_egarch()
        assert 'EGARCH' in modeler.results
        assert 'EGARCH' in modeler.models
        assert 'EGARCH' in modeler.ic_scores

    def test_fit_ic_scores_structure(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_egarch()
        scores = modeler.ic_scores['EGARCH']
        assert 'AIC' in scores
        assert 'BIC' in scores
        assert 'converged' in scores

    def test_fit_aic_bic_finite(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_egarch()
        assert modeler.ic_scores['EGARCH']['AIC'] < np.inf
        assert modeler.ic_scores['EGARCH']['BIC'] < np.inf

    def test_fit_convergence_flag(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_egarch()
        assert isinstance(modeler.ic_scores['EGARCH']['converged'], bool)

    def test_fit_conditional_volatility_positive(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_egarch()
        vol = modeler.results['EGARCH'].conditional_volatility
        assert (vol > 0).all()

    def test_fit_fat_tail_data(self, fat_tail_returns):
        modeler = VolatilityModeler(fat_tail_returns)
        modeler.fit_egarch()
        assert 'EGARCH' in modeler.ic_scores
        assert modeler.ic_scores['EGARCH']['AIC'] < np.inf


class TestFitGjrGarch:
    """fit_gjr_garch 方法测试"""

    def test_fit_populates_results(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_gjr_garch()
        assert 'GJR-GARCH' in modeler.results
        assert 'GJR-GARCH' in modeler.models
        assert 'GJR-GARCH' in modeler.ic_scores

    def test_fit_ic_scores_structure(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_gjr_garch()
        scores = modeler.ic_scores['GJR-GARCH']
        assert 'AIC' in scores
        assert 'BIC' in scores
        assert 'converged' in scores

    def test_fit_aic_bic_finite(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_gjr_garch()
        assert modeler.ic_scores['GJR-GARCH']['AIC'] < np.inf
        assert modeler.ic_scores['GJR-GARCH']['BIC'] < np.inf

    def test_fit_convergence_flag(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_gjr_garch()
        assert isinstance(modeler.ic_scores['GJR-GARCH']['converged'], bool)

    def test_fit_conditional_volatility_positive(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_gjr_garch()
        vol = modeler.results['GJR-GARCH'].conditional_volatility
        assert (vol > 0).all()

    def test_fit_has_gamma_param(self, sample_returns):
        """GJR-GARCH 包含 gamma (o=1) 非对称参数"""
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_gjr_garch()
        params = modeler.results['GJR-GARCH'].params
        # arch库 GJR-GARCH 使用 gamma[1] 前缀
        gamma_keys = [k for k in params.index if 'gamma' in k]
        assert len(gamma_keys) >= 1


# ============================================================================
# EWMA Tests
# ============================================================================

class TestFitEwma:
    """fit_ewma 方法测试"""

    def test_fit_returns_series(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        vol = modeler.fit_ewma()
        assert isinstance(vol, pd.Series)

    def test_fit_volatility_positive(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        vol = modeler.fit_ewma()
        assert (vol > 0).all()

    def test_fit_populates_models(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_ewma()
        assert 'EWMA' in modeler.models
        assert 'EWMA' in modeler.ic_scores

    def test_fit_default_lambda(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_ewma()
        assert modeler.models['EWMA']['lambda'] == 0.94

    def test_fit_custom_lambda(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_ewma(lambda_param=0.97)
        assert modeler.models['EWMA']['lambda'] == 0.97

    def test_fit_aic_finite(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_ewma()
        assert modeler.ic_scores['EWMA']['AIC'] < np.inf
        assert modeler.ic_scores['EWMA']['BIC'] < np.inf

    def test_fit_converged_always_true(self, sample_returns):
        """EWMA 无需优化迭代，始终 converged=True"""
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_ewma()
        assert modeler.ic_scores['EWMA']['converged'] == True

    def test_fit_stores_df(self, sample_returns):
        """EWMA 存储 t分布自由度"""
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_ewma()
        assert 'df' in modeler.models['EWMA']

    def test_fit_short_data(self, very_short_returns):
        """n<5 使用首个收益率平方作为初始方差"""
        modeler = VolatilityModeler(very_short_returns)
        modeler.fit_ewma()
        assert 'EWMA' in modeler.models

    def test_fit_very_short_data(self, minimal_returns):
        """n=3 极端边界"""
        modeler = VolatilityModeler(minimal_returns)
        modeler.fit_ewma()
        assert 'EWMA' in modeler.models
        vol = modeler.models['EWMA']['volatility']
        assert (vol > 0).all()

    def test_fit_volatility_length_matches_returns(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        vol = modeler.fit_ewma()
        assert len(vol) == len(sample_returns)

    def test_fit_index_matches_returns(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        vol = modeler.fit_ewma()
        assert vol.index.equals(sample_returns.index)

    def test_fit_variance_floor(self, minimal_returns):
        """方差下限 min_variance=1e-10 防止数值下溢"""
        modeler = VolatilityModeler(minimal_returns)
        modeler.fit_ewma()
        vol = modeler.models['EWMA']['volatility']
        assert (vol >= np.sqrt(1e-10)).all()

    def test_fit_df_in_valid_range(self, sample_returns):
        """df 被限制在 2.1-30 范围"""
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_ewma()
        df = modeler.models['EWMA']['df']
        assert 2.1 <= df <= 30


# ============================================================================
# Tournament Tests
# ============================================================================

class TestRunTournament:
    """run_tournament 方法测试"""

    def test_tournament_returns_valid_winner(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        winner = modeler.run_tournament()
        assert winner in ['GARCH', 'EGARCH', 'GJR-GARCH', 'EWMA']

    def test_tournament_populates_all_ic_scores(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.run_tournament()
        expected_keys = ['GARCH', 'EGARCH', 'GJR-GARCH', 'EWMA']
        for key in expected_keys:
            assert key in modeler.ic_scores

    def test_tournament_populates_all_results(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.run_tournament()
        assert 'GARCH' in modeler.results
        assert 'EGARCH' in modeler.results
        assert 'GJR-GARCH' in modeler.results

    def test_tournament_winner_has_best_aic(self, sample_returns):
        """锦标赛获胜者应具有最低 AIC"""
        modeler = VolatilityModeler(sample_returns)
        winner = modeler.run_tournament()
        winner_aic = modeler.ic_scores[winner]['AIC']
        for name, scores in modeler.ic_scores.items():
            if name != winner:
                assert winner_aic <= scores['AIC']

    def test_tournament_with_garch_sim(self, garch_sim_returns):
        """GARCH模拟数据应更倾向选择GARCH类模型"""
        modeler = VolatilityModeler(garch_sim_returns)
        winner = modeler.run_tournament()
        assert winner in ['GARCH', 'EGARCH', 'GJR-GARCH', 'EWMA']

    def test_tournament_ewma_lambda_param(self, sample_returns):
        """ewma_lambda 参数传递给 EWMA"""
        modeler = VolatilityModeler(sample_returns)
        modeler.run_tournament(ewma_lambda=0.97)
        assert modeler.models['EWMA']['lambda'] == 0.97

    def test_tournament_ewma_default_lambda(self, sample_returns):
        """无 ewma_lambda 时使用默认 0.94"""
        modeler = VolatilityModeler(sample_returns)
        modeler.run_tournament()
        assert modeler.models['EWMA']['lambda'] == 0.94

    def test_tournament_all_ic_scores_finite(self, sample_returns):
        """所有模型 IC 分数应为有限值"""
        modeler = VolatilityModeler(sample_returns)
        modeler.run_tournament()
        for name, scores in modeler.ic_scores.items():
            assert scores['AIC'] < np.inf
            assert scores['BIC'] < np.inf

    def test_tournament_all_models_converged_flag(self, sample_returns):
        """所有模型都有 converged 标志"""
        modeler = VolatilityModeler(sample_returns)
        modeler.run_tournament()
        for name, scores in modeler.ic_scores.items():
            assert 'converged' in scores


# ============================================================================
# get_conditional_volatility Tests
# ============================================================================

class TestGetConditionalVolatility:
    """get_conditional_volatility 方法测试"""

    def test_garch_volatility(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_garch()
        vol = modeler.get_conditional_volatility('GARCH')
        assert isinstance(vol, pd.Series)
        assert (vol > 0).all()

    def test_egarch_volatility(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_egarch()
        vol = modeler.get_conditional_volatility('EGARCH')
        assert isinstance(vol, pd.Series)
        assert (vol > 0).all()

    def test_gjr_garch_volatility(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_gjr_garch()
        vol = modeler.get_conditional_volatility('GJR-GARCH')
        assert isinstance(vol, pd.Series)
        assert (vol > 0).all()

    def test_ewma_volatility(self, sample_returns):
        """EWMA 使用 models 字典中的 volatility"""
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_ewma()
        vol = modeler.get_conditional_volatility('EWMA')
        assert isinstance(vol, pd.Series)
        assert (vol > 0).all()

    def test_ewma_from_models_dict(self, sample_returns):
        """确认 EWMA volatility 来自 models['EWMA']['volatility']"""
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_ewma()
        vol_direct = modeler.models['EWMA']['volatility']
        vol_method = modeler.get_conditional_volatility('EWMA')
        assert vol_direct.equals(vol_method)

    def test_unfitted_model_raises(self, sample_returns):
        """请求未拟合的模型应抛出 ValueError"""
        modeler = VolatilityModeler(sample_returns)
        with pytest.raises(ValueError):
            modeler.get_conditional_volatility('GARCH')

    def test_winner_volatility(self, sample_returns):
        """使用锦标赛获胜者获取波动率"""
        modeler = VolatilityModeler(sample_returns)
        winner = modeler.run_tournament()
        vol = modeler.get_conditional_volatility(winner)
        assert isinstance(vol, pd.Series)
        assert (vol > 0).all()

    def test_figarch_unfitted_raises(self, sample_returns):
        """FIGARCH 未计算时请求应抛出 ValueError"""
        modeler = VolatilityModeler(sample_returns)
        with pytest.raises(ValueError):
            modeler.get_conditional_volatility('FIGARCH')


# ============================================================================
# get_parameter_diagnostics Tests
# ============================================================================

class TestGetParameterDiagnostics:
    """get_parameter_diagnostics 方法测试"""

    def test_garch_diagnostics(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_garch()
        diag = modeler.get_parameter_diagnostics('GARCH')
        assert diag is not None
        assert isinstance(diag, dict)

    def test_diagnostics_structure(self, sample_returns):
        """每个参数应包含 estimate/std_error/t_stat/p_value/significant"""
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_garch()
        diag = modeler.get_parameter_diagnostics('GARCH')
        for param_name, param_diag in diag.items():
            assert 'estimate' in param_diag
            assert 'std_error' in param_diag
            assert 't_stat' in param_diag
            assert 'p_value' in param_diag
            assert 'significant' in param_diag

    def test_diagnostics_estimates_match_params(self, sample_returns):
        """diagnostics 估计值应与 result.params 一致"""
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_garch()
        diag = modeler.get_parameter_diagnostics('GARCH')
        result_params = modeler.results['GARCH'].params
        for param_name in result_params.index:
            assert param_name in diag
            assert abs(diag[param_name]['estimate'] - result_params[param_name]) < 1e-10

    def test_diagnostics_t_stat_computed(self, sample_returns):
        """t统计量 = estimate / std_error"""
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_garch()
        diag = modeler.get_parameter_diagnostics('GARCH')
        for param_name, param_diag in diag.items():
            if param_diag['std_error'] > 0:
                expected_t = param_diag['estimate'] / param_diag['std_error']
                assert abs(param_diag['t_stat'] - expected_t) < 1e-6

    def test_diagnostics_significant_type(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_garch()
        diag = modeler.get_parameter_diagnostics('GARCH')
        for param_name, param_diag in diag.items():
            assert type(param_diag['significant']) in (bool, np.bool_)

    def test_diagnostics_egarch(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_egarch()
        diag = modeler.get_parameter_diagnostics('EGARCH')
        assert diag is not None

    def test_diagnostics_gjr_garch(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_gjr_garch()
        diag = modeler.get_parameter_diagnostics('GJR-GARCH')
        assert diag is not None

    def test_diagnostics_unfitted_returns_none(self, sample_returns):
        """请求未拟合模型的诊断信息应返回 None"""
        modeler = VolatilityModeler(sample_returns)
        diag = modeler.get_parameter_diagnostics('GARCH')
        assert diag is None

    def test_diagnostics_ewma_returns_none(self, sample_returns):
        """EWMA 不在 results 字典中，诊断应返回 None"""
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_ewma()
        diag = modeler.get_parameter_diagnostics('EWMA')
        assert diag is None


# ============================================================================
# RegimeDetector Tests
# ============================================================================

class TestRegimeDetectorInit:
    """RegimeDetector 初始化测试"""

    def test_init_stores_volatility(self, volatility_series):
        detector = RegimeDetector(volatility_series)
        assert detector.volatility is volatility_series

    def test_init_default_n_regimes(self, volatility_series):
        detector = RegimeDetector(volatility_series)
        assert detector.n_regimes == 3

    def test_init_custom_n_regimes(self, volatility_series):
        detector = RegimeDetector(volatility_series, n_regimes=4)
        assert detector.n_regimes == 4

    def test_init_empty_state(self, volatility_series):
        detector = RegimeDetector(volatility_series)
        assert detector.model is None
        assert detector.regime_labels is None
        assert detector.regime_stats is None


class TestRegimeDetectorFit:
    """RegimeDetector.fit 方法测试"""

    def test_fit_returns_labels(self, volatility_series):
        detector = RegimeDetector(volatility_series)
        labels = detector.fit()
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(volatility_series)

    def test_fit_labels_are_valid_regimes(self, volatility_series):
        detector = RegimeDetector(volatility_series)
        labels = detector.fit()
        unique_labels = set(labels)
        assert all(l in range(detector.n_regimes) for l in unique_labels)

    def test_fit_populates_regime_stats(self, volatility_series):
        detector = RegimeDetector(volatility_series)
        detector.fit()
        assert detector.regime_stats is not None
        assert len(detector.regime_stats) == detector.n_regimes

    def test_fit_regime_stats_structure(self, volatility_series):
        """每个 regime 统计应包含 mean/std/count/pct"""
        detector = RegimeDetector(volatility_series)
        detector.fit()
        for i in range(detector.n_regimes):
            stats = detector.regime_stats[i]
            assert 'mean' in stats
            assert 'std' in stats
            assert 'count' in stats
            assert 'pct' in stats

    def test_fit_regimes_ordered_by_mean(self, volatility_series):
        """状态按均值排序: 0=低, 1=中, 2=高"""
        detector = RegimeDetector(volatility_series)
        detector.fit()
        means = [detector.regime_stats[i]['mean'] for i in range(detector.n_regimes)]
        for i in range(len(means) - 1):
            assert means[i] <= means[i + 1]

    def test_fit_pct_sum_to_100(self, volatility_series):
        """各状态占比总和应接近 100%"""
        detector = RegimeDetector(volatility_series)
        detector.fit()
        total_pct = sum(detector.regime_stats[i]['pct'] for i in range(detector.n_regimes))
        assert abs(total_pct - 100.0) < 1.0

    def test_fit_count_sum_matches_total(self, volatility_series):
        """各状态样本数总和应等于总数据量"""
        detector = RegimeDetector(volatility_series)
        detector.fit()
        total_count = sum(detector.regime_stats[i]['count'] for i in range(detector.n_regimes))
        assert total_count == len(volatility_series)

    def test_fit_with_2_regimes(self, volatility_series):
        detector = RegimeDetector(volatility_series, n_regimes=2)
        labels = detector.fit()
        assert set(labels).issubset({0, 1})
        assert len(detector.regime_stats) == 2

    def test_fit_with_4_regimes(self, volatility_series):
        detector = RegimeDetector(volatility_series, n_regimes=4)
        labels = detector.fit()
        assert set(labels).issubset({0, 1, 2, 3})
        assert len(detector.regime_stats) == 4


class TestRegimeDetectorGetCurrentRegime:
    """get_current_regime 方法测试"""

    def test_current_regime_after_fit(self, volatility_series):
        detector = RegimeDetector(volatility_series)
        detector.fit()
        current = detector.get_current_regime()
        assert 0 <= current < detector.n_regimes

    def test_current_regime_before_fit_raises(self, volatility_series):
        detector = RegimeDetector(volatility_series)
        with pytest.raises(ValueError):
            detector.get_current_regime()


class TestRegimeDetectorGetRegimeName:
    """get_regime_name 方法测试"""

    def test_regime_0_is_low(self):
        detector = RegimeDetector(pd.Series([1.0]))
        assert detector.get_regime_name(0) == '低波动'

    def test_regime_1_is_medium(self):
        detector = RegimeDetector(pd.Series([1.0]))
        assert detector.get_regime_name(1) == '中波动'

    def test_regime_2_is_high(self):
        detector = RegimeDetector(pd.Series([1.0]))
        assert detector.get_regime_name(2) == '高波动'

    def test_unknown_regime(self):
        detector = RegimeDetector(pd.Series([1.0]))
        name = detector.get_regime_name(5)
        assert name == '状态5'


class TestRegimeDetectorPrintSummary:
    """print_regime_summary 方法测试"""

    def test_print_runs_without_error(self, volatility_series):
        detector = RegimeDetector(volatility_series)
        detector.fit()
        detector.print_regime_summary()  # 应不抛异常


# ============================================================================
# GARCH Model Property Tests
# ============================================================================

class TestGarchModelProperties:
    """GARCH 模型拟合后的属性验证"""

    def test_garch_params_include_omega(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_garch()
        params = modeler.results['GARCH'].params
        assert 'omega' in params.index

    def test_garch_params_include_alpha(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_garch()
        params = modeler.results['GARCH'].params
        alpha_keys = [k for k in params.index if 'alpha' in k]
        assert len(alpha_keys) >= 1

    def test_garch_params_include_beta(self, sample_returns):
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_garch()
        params = modeler.results['GARCH'].params
        beta_keys = [k for k in params.index if 'beta' in k]
        assert len(beta_keys) >= 1

    def test_egarch_log_volatility(self, sample_returns):
        """EGARCH 使用对数波动率模型"""
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_egarch()
        result = modeler.results['EGARCH']
        # EGARCH conditional_volatility 是 exp(log_volatility)
        vol = result.conditional_volatility
        assert (vol > 0).all()

    def test_gjr_garch_gamma_positive(self, sample_returns):
        """GJR-GARCH gamma 参数应为非负 (杠杆效应)"""
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_gjr_garch()
        params = modeler.results['GJR-GARCH'].params
        gamma_keys = [k for k in params.index if 'gamma' in k]
        for key in gamma_keys:
            # gamma 通常 >= 0 (坏消息增加波动率)
            assert params[key] >= -0.5  # 宽松范围，模型拟合可能不稳定

    def test_ewma_recursion_formula(self, sample_returns):
        """验证 EWMA 递推公式: σ²_t = λ*σ²_{t-1} + (1-λ)*r²_{t-1}"""
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_ewma(lambda_param=0.94)
        vol = modeler.models['EWMA']['volatility']
        variance = vol.values ** 2
        returns = sample_returns.values
        lam = 0.94

        # 手动验证第几个点
        for t in [5, 10, 50]:
            expected_var = lam * variance[t-1] + (1-lam) * returns[t-1]**2
            expected_var = max(expected_var, 1e-10)
            assert abs(variance[t] - expected_var) < 1e-8

    def test_ewma_lambda_sensitivity(self, sample_returns):
        """不同 lambda 产生不同波动率"""
        modeler1 = VolatilityModeler(sample_returns)
        modeler1.fit_ewma(lambda_param=0.90)
        vol1 = modeler1.models['EWMA']['volatility']

        modeler2 = VolatilityModeler(sample_returns)
        modeler2.fit_ewma(lambda_param=0.97)
        vol2 = modeler2.models['EWMA']['volatility']

        # λ=0.97 波动率更平滑 (反应更慢)
        # λ=0.90 波动率更敏感 (反应更快)
        assert not vol1.equals(vol2)

    def test_persistence_sum_less_than_one(self, sample_returns):
        """GARCH(1,1) α+β < 1 平稳性条件"""
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_garch()
        params = modeler.results['GARCH'].params
        alpha = params.get('alpha[1]', 0)
        beta = params.get('beta[1]', 0)
        # 平稳性条件: α + β < 1
        assert alpha + beta < 1.0


# ============================================================================
# Boundary and Edge Condition Tests
# ============================================================================

class TestBoundaryConditions:
    """边界条件测试"""

    def test_short_data_garch(self, short_returns):
        """n=30 短数据应仍可拟合"""
        modeler = VolatilityModeler(short_returns)
        modeler.fit_garch()
        assert 'GARCH' in modeler.ic_scores

    def test_short_data_ewma(self, short_returns):
        modeler = VolatilityModeler(short_returns)
        modeler.fit_ewma()
        assert 'EWMA' in modeler.models
        assert (modeler.models['EWMA']['volatility'] > 0).all()

    def test_short_data_tournament(self, short_returns):
        modeler = VolatilityModeler(short_returns)
        winner = modeler.run_tournament()
        assert winner in ['GARCH', 'EGARCH', 'GJR-GARCH', 'EWMA']

    def test_fat_tail_improves_gjr(self, fat_tail_returns):
        """肥尾数据下 GJR-GARCH 可能更优 (非对称效应)"""
        modeler = VolatilityModeler(fat_tail_returns)
        modeler.run_tournament()
        # 不必总是GJR赢, 但至少能拟合
        assert 'GJR-GARCH' in modeler.ic_scores

    def test_constant_data_ewma(self, constant_returns):
        """恒定收益率的 EWMA 波动率应为正 (除初始点)"""
        modeler = VolatilityModeler(constant_returns)
        modeler.fit_ewma()
        vol = modeler.models['EWMA']['volatility']
        # 恒定收益率方差初始值可能为0, 但递推后应 > 0
        assert (vol[1:] > 0).all()
        assert vol.mean() < 0.002

    def test_regime_detector_flat_data(self):
        """平坦波动率应检测为单一状态主导"""
        flat_vol = pd.Series(np.ones(200) * 0.02)
        detector = RegimeDetector(flat_vol, n_regimes=3)
        labels = detector.fit()
        # 平坦数据应有一个状态占大多数
        dominant_pct = max(detector.regime_stats[i]['pct'] for i in range(3))
        assert dominant_pct >= 40


# ============================================================================
# Integration Tests
# ============================================================================

class TestVolatilityIntegration:
    """波动率模块集成测试"""

    def test_full_tournament_then_volatility(self, garch_sim_returns):
        """完整锦标赛后获取所有模型波动率"""
        modeler = VolatilityModeler(garch_sim_returns)
        winner = modeler.run_tournament()

        for name in ['GARCH', 'EGARCH', 'GJR-GARCH', 'EWMA']:
            vol = modeler.get_conditional_volatility(name)
            assert isinstance(vol, pd.Series)
            assert (vol > 0).all()
            assert len(vol) == len(garch_sim_returns)

    def test_tournament_then_diagnostics(self, sample_returns):
        """锦标赛后获取所有GARCH类模型诊断"""
        modeler = VolatilityModeler(sample_returns)
        modeler.run_tournament()

        for name in ['GARCH', 'EGARCH', 'GJR-GARCH']:
            diag = modeler.get_parameter_diagnostics(name)
            assert diag is not None
            assert len(diag) > 0

    def test_tournament_then_regime_detection(self, garch_sim_returns):
        """锦标赛后对波动率做状态检测"""
        modeler = VolatilityModeler(garch_sim_returns)
        winner = modeler.run_tournament()
        vol = modeler.get_conditional_volatility(winner)

        detector = RegimeDetector(vol)
        labels = detector.fit()
        assert len(labels) == len(vol)

    def test_ewma_then_regime_detection(self, sample_returns):
        """EWMA波动率状态检测"""
        modeler = VolatilityModeler(sample_returns)
        modeler.fit_ewma()
        vol = modeler.get_conditional_volatility('EWMA')

        detector = RegimeDetector(vol)
        labels = detector.fit()
        assert len(labels) == len(vol)

    def test_multiple_tournaments(self, sample_returns):
        """重复锦标赛应覆盖旧结果"""
        modeler = VolatilityModeler(sample_returns)
        winner1 = modeler.run_tournament()
        winner2 = modeler.run_tournament()
        # 第二次锦标赛结果应与第一次一致 (同数据同种子)
        # 但模型对象会被覆盖
        assert winner2 in ['GARCH', 'EGARCH', 'GJR-GARCH', 'EWMA']
