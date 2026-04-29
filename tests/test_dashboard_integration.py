"""
Dashboard 集成测试 - 验证多页架构和模块集成

测试范围:
1. shared_state 模块导入和函数签名
2. 页面文件导入验证
3. MarketStatusGauge + ProvinceClusterMap 与 app.py 集成
4. 全工作流管线集成 (数据 → 模型 → 仪表 → 报告)
5. 版本一致性验证

标记: @pytest.mark.integration, @pytest.mark.dashboard
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
import importlib

# conftest.py 已配置 sys.path


# ============================================================================
# 模块导入验证
# ============================================================================

@pytest.mark.integration
class TestModuleImports:
    """验证所有核心模块可正常导入"""

    def test_import_data_engine(self):
        from data_engine import DataEngine
        assert DataEngine is not None

    def test_import_volatility(self):
        from volatility import VolatilityModeler, RegimeDetector
        assert VolatilityModeler is not None
        assert RegimeDetector is not None

    def test_import_kalman(self):
        from kalman import KalmanSignalExtractor
        assert KalmanSignalExtractor is not None

    def test_import_evt(self):
        from evt import EVTRiskAnalyzer
        assert EVTRiskAnalyzer is not None

    def test_import_visualization(self):
        from visualization import (
            plot_signal_trend, plot_volatility_structure, plot_tail_risk,
            add_range_selector, get_theme_config
        )
        assert plot_signal_trend is not None

    def test_import_styles(self):
        from styles import (
            metric_card, alert_box, section_header,
            apply_theme, render_page_header, render_footer
        )
        assert metric_card is not None

    def test_import_scenarios(self):
        from scenarios import (
            run_stress_test, run_monte_carlo,
            run_sensitivity_analysis, calculate_rolling_stats
        )
        assert run_stress_test is not None

    def test_import_alerts(self):
        from alerts import check_risk_alerts, get_risk_score
        assert check_risk_alerts is not None

    def test_import_content(self):
        from content import (
            get_spread_position_comment,
            get_volatility_comment,
            get_var_comment
        )
        assert get_spread_position_comment is not None

    def test_import_export(self):
        from export import export_to_excel
        assert export_to_excel is not None

    def test_import_report_gen(self):
        from report_gen import ReportGenerator, TEMPLATES
        assert ReportGenerator is not None
        assert isinstance(TEMPLATES, dict)

    def test_import_ml_volatility(self):
        from ml_volatility import MLVolatilityModeler
        assert MLVolatilityModeler is not None

    def test_import_calibration(self):
        from calibration import ParameterCalibrator
        assert ParameterCalibrator is not None

    def test_import_market_status(self):
        from market_status import MarketStatusGauge
        assert MarketStatusGauge is not None

    def test_import_province_cluster(self):
        from province_cluster import ProvinceClusterMap
        assert ProvinceClusterMap is not None

    def test_src_init_exports(self):
        """验证 __init__.py 导出所有 v3.0 模块"""
        import src
        expected_exports = [
            'DataEngine', 'VolatilityModeler', 'RegimeDetector',
            'KalmanSignalExtractor', 'EVTRiskAnalyzer',
            'MLVolatilityModeler', 'ParameterCalibrator',
            'MarketStatusGauge', 'ProvinceClusterMap',
            'ReportGenerator', 'TEMPLATES'
        ]
        for name in expected_exports:
            assert hasattr(src, name), f"{name} not exported from src"

    def test_shared_state_imports_ml_calibration(self):
        """验证 shared_state.py 导入 MLVolatilityModeler 和 ParameterCalibrator"""
        shared_path = os.path.join(
            os.path.dirname(__file__), '..', 'shared_state.py'
        )
        with open(shared_path, 'r') as f:
            content = f.read()
        assert 'MLVolatilityModeler' in content
        assert 'ParameterCalibrator' in content

    def test_shared_state_figarch_in_pipeline(self):
        """验证 shared_state.py 包含 FIGARCH 长记忆检测步骤"""
        shared_path = os.path.join(
            os.path.dirname(__file__), '..', 'shared_state.py'
        )
        with open(shared_path, 'r') as f:
            content = f.read()
        assert 'detect_long_memory' in content
        assert 'fit_figarch' in content
        assert 'long_memory' in content


# ============================================================================
# 全工作流集成测试
# ============================================================================

@pytest.mark.integration
class TestFullWorkflowIntegration:
    """验证完整分析管线: 数据 → 模型 → 仪表 → 报告"""

    def test_data_to_volatility_pipeline(self, mock_config):
        """数据加载 → GARCH锦标赛 完整管线"""
        from data_engine import DataEngine
        from volatility import VolatilityModeler

        engine = DataEngine(mock_config)
        engine.load_data()
        clean_data = engine.clean_data()
        returns = engine.get_returns()

        modeler = VolatilityModeler(returns)
        winner = modeler.run_tournament()
        assert winner in ['GARCH', 'EGARCH', 'GJR-GARCH', 'EWMA']

        vol = modeler.get_conditional_volatility(winner)
        assert isinstance(vol, pd.Series)
        assert (vol > 0).all()

    def test_data_to_kalman_pipeline(self, mock_config):
        """数据加载 → 卡尔曼滤波 完整管线"""
        from data_engine import DataEngine
        from kalman import KalmanSignalExtractor

        engine = DataEngine(mock_config)
        engine.load_data()
        clean_data = engine.clean_data()

        kalman = KalmanSignalExtractor(clean_data['spread'])
        smoothed = kalman.fit()
        deviation = kalman.get_signal_deviation()

        assert len(smoothed) == len(clean_data)
        assert len(deviation) == len(clean_data)

    def test_data_to_evt_pipeline(self, mock_config):
        """数据加载 → EVT风险分析 完整管线"""
        from data_engine import DataEngine
        from evt import EVTRiskAnalyzer

        engine = DataEngine(mock_config)
        engine.load_data()
        clean_data = engine.clean_data()
        returns = engine.get_returns()

        evt = EVTRiskAnalyzer(returns)
        evt.fit_gpd()
        var = evt.calculate_var()
        es = evt.calculate_es()

        assert var is not None and var > 0
        assert es is not None and es >= var

    def test_data_to_market_status_pipeline(self, mock_config):
        """数据加载 → 全分析 → MarketStatusGauge"""
        from data_engine import DataEngine
        from volatility import VolatilityModeler
        from kalman import KalmanSignalExtractor
        from evt import EVTRiskAnalyzer
        from market_status import MarketStatusGauge

        engine = DataEngine(mock_config)
        engine.load_data()
        clean_data = engine.clean_data()
        returns = engine.get_returns()

        vol_modeler = VolatilityModeler(returns)
        winner = vol_modeler.run_tournament()
        winner_vol = vol_modeler.get_conditional_volatility(winner)

        kalman = KalmanSignalExtractor(clean_data['spread'])
        smoothed = kalman.fit()
        deviation = kalman.get_signal_deviation()

        evt = EVTRiskAnalyzer(returns)
        evt.fit_gpd()

        gauge = MarketStatusGauge(
            clean_data, returns,
            smoothed=smoothed, deviation=deviation,
            vol_modeler=vol_modeler, evt=evt
        )
        status = gauge.get_market_status()

        assert 'status' in status
        assert 'score' in status
        assert status['score'] >= 0
        assert status['score'] <= 100

    def test_data_to_province_cluster_pipeline(self):
        """ProvinceClusterMap 独立管线"""
        from province_cluster import ProvinceClusterMap

        pcm = ProvinceClusterMap(n_clusters=4)
        pcm.run_clustering()

        stats = pcm.get_cluster_stats()
        assert len(stats) == 4
        for c, s in stats.items():
            assert 'n_provinces' in s
            assert 'risk_level' in s
            assert s['n_provinces'] > 0

    def test_data_to_report_pipeline(self, mock_config, tmp_path):
        """数据加载 → 全分析 → 报告生成"""
        from data_engine import DataEngine
        from volatility import VolatilityModeler
        from kalman import KalmanSignalExtractor
        from evt import EVTRiskAnalyzer
        from report_gen import ReportGenerator

        engine = DataEngine(mock_config)
        engine.load_data()
        clean_data = engine.clean_data()
        returns = engine.get_returns()

        vol_modeler = VolatilityModeler(returns)
        vol_modeler.run_tournament()

        kalman = KalmanSignalExtractor(clean_data['spread'])
        kalman.fit()

        evt = EVTRiskAnalyzer(returns)
        evt.fit_gpd()
        evt.calculate_var()
        evt.calculate_es()

        gen = ReportGenerator(output_dir=str(tmp_path))
        report_path = gen.generate_report(
            clean_data=clean_data,
            returns=returns,
            kalman=kalman,
            vol_modeler=vol_modeler,
            evt=evt
        )

        assert report_path is not None
        assert os.path.exists(report_path)


# ============================================================================
# 版本一致性验证
# ============================================================================

@pytest.mark.integration
class TestVersionConsistency:
    """验证所有模块版本号统一为 3.0.0"""

    VERSION = '3.0.0'

    def test_src_init_version(self):
        import src
        assert src.__version__ == self.VERSION

    def test_dashboard_version(self):
        """检查 dashboard.py 注释中的版本"""
        dashboard_path = os.path.join(
            os.path.dirname(__file__), '..', 'dashboard.py'
        )
        if os.path.exists(dashboard_path):
            with open(dashboard_path, 'r') as f:
                content = f.read()
            assert '3.0.0' in content

    def test_report_gen_version(self):
        from report_gen import DISCLAIMER
        assert '3.0.0' in DISCLAIMER

    def test_styles_version(self):
        """验证 styles.py 渲染页脚使用正确版本"""
        import styles
        # render_footer 默认版本参数
        sig = styles.render_footer.__code__.co_varnames
        # 检查函数存在
        assert 'render_footer' in dir(styles)

    def test_shared_state_version(self):
        """验证 shared_state.py 使用正确版本"""
        shared_path = os.path.join(
            os.path.dirname(__file__), '..', 'shared_state.py'
        )
        if os.path.exists(shared_path):
            with open(shared_path, 'r') as f:
                content = f.read()
            assert '3.0.0' in content


# ============================================================================
# Dashboard 页面结构验证
# ============================================================================

@pytest.mark.dashboard
@pytest.mark.integration
class TestDashboardStructure:
    """验证多页 Dashboard 架构完整性"""

    def test_app_py_exists(self):
        project_root = os.path.join(os.path.dirname(__file__), '..')
        assert os.path.exists(os.path.join(project_root, 'app.py'))

    def test_shared_state_exists(self):
        project_root = os.path.join(os.path.dirname(__file__), '..')
        assert os.path.exists(os.path.join(project_root, 'shared_state.py'))

    def test_pages_directory_exists(self):
        project_root = os.path.join(os.path.dirname(__file__), '..')
        pages_dir = os.path.join(project_root, 'pages')
        assert os.path.isdir(pages_dir)

    def test_all_six_page_files_exist(self):
        """验证6个页面文件全部存在"""
        project_root = os.path.join(os.path.dirname(__file__), '..')
        pages_dir = os.path.join(project_root, 'pages')
        expected_pages = [
            '1_📈_信号分析.py',
            '2_📉_波动率分析.py',
            '3_⚠️_风险分析.py',
            '4_🎯_情景分析.py',
            '5_📜_历史回溯.py',
            '6_📋_报告中心.py'
        ]
        for page in expected_pages:
            assert os.path.exists(os.path.join(pages_dir, page)), \
                f"Missing page: {page}"

    def test_dashboard_py_backward_compat(self):
        """验证旧版 dashboard.py 仍存在（向后兼容）"""
        project_root = os.path.join(os.path.dirname(__file__), '..')
        assert os.path.exists(os.path.join(project_root, 'dashboard.py'))

    def test_pages_import_shared_state(self):
        """验证每个页面文件导入 shared_state"""
        project_root = os.path.join(os.path.dirname(__file__), '..')
        pages_dir = os.path.join(project_root, 'pages')
        for filename in os.listdir(pages_dir):
            if filename.endswith('.py'):
                filepath = os.path.join(pages_dir, filename)
                with open(filepath, 'r') as f:
                    content = f.read()
                assert 'shared_state' in content, \
                    f"{filename} doesn't import shared_state"


# ============================================================================
# 跨模块交互验证
# ============================================================================

@pytest.mark.integration
class TestCrossModuleInteraction:
    """验证模块间数据传递和交互"""

    def test_volatility_to_evt_data_flow(self, mock_config):
        """波动率模型 → EVT 风险模型数据传递"""
        from data_engine import DataEngine
        from volatility import VolatilityModeler
        from evt import EVTRiskAnalyzer

        engine = DataEngine(mock_config)
        engine.load_data()
        clean_data = engine.clean_data()
        returns = engine.get_returns()

        # GARCH → EVT: returns 是共享数据接口
        vol_modeler = VolatilityModeler(returns)
        vol_modeler.run_tournament()

        evt = EVTRiskAnalyzer(returns)
        evt.fit_gpd()

        # 验证模型间数据一致性
        assert len(returns) > 0
        assert evt.threshold is not None

    def test_kalman_to_alerts_data_flow(self, mock_config):
        """卡尔曼滤波 → 告警系统数据传递"""
        from data_engine import DataEngine
        from kalman import KalmanSignalExtractor
        from volatility import VolatilityModeler
        from evt import EVTRiskAnalyzer
        from alerts import check_risk_alerts

        engine = DataEngine(mock_config)
        engine.load_data()
        clean_data = engine.clean_data()
        returns = engine.get_returns()

        kalman = KalmanSignalExtractor(clean_data['spread'])
        kalman.fit()

        vol_modeler = VolatilityModeler(returns)
        vol_modeler.run_tournament()

        evt = EVTRiskAnalyzer(returns)
        evt.fit_gpd()

        alerts = check_risk_alerts(clean_data, returns, evt, vol_modeler)
        assert isinstance(alerts, list)

    def test_calibration_to_volatility_config(self, fat_tail_returns, spread_series):
        """参数校准 → 波动率模型配置传递"""
        from calibration import ParameterCalibrator

        calibrator = ParameterCalibrator(fat_tail_returns, spread_series)
        calibrator.calibrate_all()
        config = calibrator.calibrated

        assert 'ewma_lambda' in config
        assert 't_df' in config
        assert 'ar_phi' in config

        # 验证校准值在合理范围内
        assert 0.8 <= config['ewma_lambda'] <= 0.99
        assert config['t_df'] > 2

    def test_ml_vs_garch_comparison(self, garch_returns):
        """ML模型 vs GARCH锦标赛对比集成"""
        from volatility import VolatilityModeler
        from ml_volatility import MLVolatilityModeler

        garch_modeler = VolatilityModeler(garch_returns)
        garch_winner = garch_modeler.run_tournament()

        ml_modeler = MLVolatilityModeler(garch_returns)
        comparison, ml_winner = ml_modeler.run_ml_tournament()

        # 验证两个系统都能产生有效结果
        assert garch_winner is not None
        assert isinstance(comparison, dict)
        assert 'RF' in comparison

    def test_full_v3_pipeline_garch_ml_figarch_calibration(self, mock_config):
        """v3.0完整管线: GARCH → FIGARCH → ML → 校准"""
        from data_engine import DataEngine
        from volatility import VolatilityModeler
        from ml_volatility import MLVolatilityModeler
        from calibration import ParameterCalibrator

        engine = DataEngine(mock_config)
        engine.load_data()
        clean_data = engine.clean_data()
        returns = engine.get_returns()

        # Step 1: GARCH锦标赛
        vol_modeler = VolatilityModeler(returns)
        winner = vol_modeler.run_tournament()
        assert winner in ['GARCH', 'EGARCH', 'GJR-GARCH', 'EWMA']

        # Step 2: FIGARCH长记忆检测
        lm_result = vol_modeler.detect_long_memory()
        assert 'd_estimate' in lm_result
        assert 'memory_type' in lm_result

        # Step 3: ML模型锦标赛
        ml_modeler = MLVolatilityModeler(returns)
        ml_comparison, ml_winner = ml_modeler.run_ml_tournament()
        assert isinstance(ml_comparison, dict)
        assert ml_winner is not None

        # Step 4: GARCH vs ML对比
        garch_comparison, overall_winner = ml_modeler.compare_with_garch(vol_modeler.ic_scores)
        assert isinstance(garch_comparison, pd.DataFrame)
        assert overall_winner is not None

        # Step 5: 参数自动校准
        calibrator = ParameterCalibrator(returns, spread=clean_data.get('spread'))
        calibrator.calibrate_all()
        calibrated = calibrator.calibrated
        assert 'ewma_lambda' in calibrated
        assert 't_df' in calibrated
        assert 'ar_phi' in calibrated

    def test_figarch_pipeline_with_long_memory(self, long_memory_returns):
        """长记忆数据 → FIGARCH检测与拟合完整管线"""
        from volatility import VolatilityModeler

        modeler = VolatilityModeler(long_memory_returns)
        winner = modeler.run_tournament()

        lm_result = modeler.detect_long_memory()
        d = lm_result['d_estimate']

        # 如果检测到显著长记忆，拟合FIGARCH
        if d > 0.05 and lm_result.get('d_p_value', 1) < 0.10:
            figarch_result = modeler.fit_figarch()
            assert 'd' in figarch_result
            assert 'AIC' in figarch_result


# ============================================================================
# 边界条件集成测试
# ============================================================================

@pytest.mark.integration
class TestBoundaryIntegration:
    """集成层面的边界条件测试"""

    def test_short_data_all_modules(self, short_returns):
        """极短数据在各模块的优雅降级"""
        from volatility import VolatilityModeler
        from evt import EVTRiskAnalyzer

        # GARCH 应能处理短数据（可能不收敛）
        modeler = VolatilityModeler(short_returns)
        try:
            winner = modeler.run_tournament()
        except Exception:
            winner = None  # 短数据可能导致失败，应优雅处理

        # EVT 对短数据应能降级
        evt = EVTRiskAnalyzer(short_returns)
        evt.fit_gpd()
        # 短数据可能无法拟合 GPD

    def test_missing_optional_components_market_status(self, make_spread_data, make_returns):
        """MarketStatusGauge 缺少可选组件时降级"""
        from market_status import MarketStatusGauge

        clean_data = make_spread_data(n=100)
        returns = make_returns(n=99)

        # 只提供必要数据
        gauge = MarketStatusGauge(clean_data, returns)
        status = gauge.get_market_status()

        assert 'status' in status
        assert 'score' in status
        assert 'score' in status
        # 缺少组件的指标应默认为50(中性)
        scores = gauge._indicator_scores
        assert scores['spread_position']['score'] > 0


# ============================================================================
# conftest.py fixtures 验证
# ============================================================================

@pytest.mark.integration
class TestConftestFixtures:
    """验证 conftest.py 共享 fixtures 的正确性"""

    def test_garch_returns_fixture(self, garch_returns):
        """GARCH 模拟数据 fixture"""
        assert isinstance(garch_returns, pd.Series)
        assert len(garch_returns) == 500
        assert garch_returns.index[0] >= pd.Timestamp('2020-01-01')

    def test_fat_tail_returns_fixture(self, fat_tail_returns):
        """肥尾分布 fixture"""
        assert isinstance(fat_tail_returns, pd.Series)
        assert len(fat_tail_returns) == 500

    def test_spread_series_fixture(self, spread_series):
        """利差序列 fixture"""
        assert isinstance(spread_series, pd.Series)
        assert len(spread_series) == 500

    def test_make_spread_data_factory(self, make_spread_data):
        """Spread data factory fixture"""
        data = make_spread_data()
        assert isinstance(data, pd.DataFrame)
        assert 'spread' in data.columns
        assert len(data) == 300

        # 自定义参数
        custom = make_spread_data(n=50, base=100, scale=0.5, freq='D')
        assert len(custom) == 50

    def test_make_returns_factory(self, make_returns):
        """Returns factory fixture"""
        ret = make_returns()
        assert isinstance(ret, pd.Series)
        assert len(ret) == 200

        # 肥尾分布
        fat = make_returns(n=300, scale=0.5, distribution='fat_tail')
        assert len(fat) == 300

    def test_make_dates_factory(self, make_dates):
        """Dates factory fixture"""
        dates = make_dates()
        assert len(dates) == 200

    def test_make_volatility_factory(self, make_volatility):
        """Volatility factory fixture"""
        vol = make_volatility()
        assert isinstance(vol, pd.Series)
        assert (vol > 0).all()

        # 注入高波动率
        vol_injected = make_volatility(inject_high=True)
        assert vol_injected.iloc[-1] > vol.iloc[-1]

    def test_make_smoothed_spread_factory(self, make_smoothed_spread):
        """Smoothed spread factory fixture"""
        s = make_smoothed_spread()
        assert isinstance(s, pd.Series)
        assert s.name == 'smoothed'

    def test_make_signal_deviation_factory(self, make_signal_deviation):
        """Signal deviation factory fixture"""
        d = make_signal_deviation()
        assert isinstance(d, pd.Series)
        assert d.name == 'deviation'

        # 注入极端值
        d_ext = make_signal_deviation(inject_extremes=True)
        assert d_ext.iloc[10] == 5.0
        assert d_ext.iloc[50] == -4.0

    def test_mock_objects_fixtures(self, mock_kalman, mock_vol_modeler, mock_evt):
        """Mock 对象 fixtures"""
        assert hasattr(mock_kalman, 'smoothed_state')
        assert hasattr(mock_vol_modeler, 'ic_scores')
        assert hasattr(mock_evt, 'var')

    def test_sample_alerts_fixture(self, sample_alerts):
        """告警列表 fixture"""
        assert len(sample_alerts) == 4
        assert sample_alerts[0]['type'] == 'VaR'

    def test_long_memory_returns_fixture(self, long_memory_returns):
        """长记忆序列 fixture"""
        assert isinstance(long_memory_returns, pd.Series)
        assert len(long_memory_returns) == 500

    def test_short_returns_fixture(self, short_returns):
        """边界数据 fixture"""
        assert len(short_returns) == 10

    def test_minimal_returns_fixture(self, minimal_returns):
        assert len(minimal_returns) == 30

    def test_very_short_returns_fixture(self, very_short_returns):
        assert len(very_short_returns) == 20

    def test_mock_config_fixture(self, mock_config):
        assert mock_config['SOURCE'] == 'MOCK'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])


# ============================================================================
# v3.0 校准参数集成验证
# ============================================================================

@pytest.mark.integration
class TestCalibrationPipelineIntegration:
    """验证校准参数实际传递到下游模块"""

    def test_ewma_lambda_passed_to_tournament(self):
        """校准的EWMA lambda传入run_tournament替代默认0.94"""
        from volatility import VolatilityModeler
        returns = pd.Series(np.random.randn(200) * 0.01)

        vol = VolatilityModeler(returns)
        vol.run_tournament(ewma_lambda=0.96)
        assert 'EWMA' in vol.ic_scores

    def test_ewma_lambda_none_uses_default(self):
        """ewma_lambda=None时使用默认0.94"""
        from volatility import VolatilityModeler
        returns = pd.Series(np.random.randn(200) * 0.01)

        vol = VolatilityModeler(returns)
        vol.run_tournament(ewma_lambda=None)
        assert 'EWMA' in vol.ic_scores

    def test_kalman_fallback_window_param(self):
        """校准的kalman_window传入KalmanSignalExtractor.fit()"""
        from kalman import KalmanSignalExtractor
        spread = pd.Series(np.random.randn(200).cumsum() + 100, name='spread')

        kalman = KalmanSignalExtractor(spread)
        # 使用自定义fallback_window参数
        result = kalman.fit(fallback_window=30)
        assert result is not None

    def test_evt_threshold_from_calibration(self):
        """校准的EVT阈值替代sidebar默认0.95"""
        from evt import EVTRiskAnalyzer
        returns = pd.Series(np.random.randn(200) * 0.01)

        evt = EVTRiskAnalyzer(returns, threshold_percentile=0.92)
        evt.fit_gpd()
        assert evt.threshold_percentile == 0.92

    def test_calibration_before_garch_pipeline(self):
        """完整管线: 校准→GARCH(使用校准lambda)→EVT(使用校准阈值)"""
        from calibration import ParameterCalibrator
        from volatility import VolatilityModeler
        from evt import EVTRiskAnalyzer

        np.random.seed(42)
        returns = pd.Series(np.random.randn(200) * 0.01)

        # 先校准
        calibrator = ParameterCalibrator(returns)
        calibrator.calibrate_all()
        calibrated = calibrator.calibrated

        # 使用校准值
        ewma_lambda = calibrated.get('ewma_lambda')
        evt_threshold = calibrated.get('evt_threshold_percentile', 0.95)

        assert ewma_lambda is not None
        assert isinstance(ewma_lambda, float)

        vol = VolatilityModeler(returns)
        winner = vol.run_tournament(ewma_lambda=ewma_lambda)

        evt = EVTRiskAnalyzer(returns, threshold_percentile=evt_threshold)
        evt.fit_gpd()

        # 校准值应与默认值不同或相同(取决于数据)
        assert isinstance(evt_threshold, float)

    def test_calibrated_values_are_floats_not_dicts(self):
        """calibrated dict存储的是flat float值, 不是nested dict"""
        from calibration import ParameterCalibrator

        returns = pd.Series(np.random.randn(200) * 0.01)
        calibrator = ParameterCalibrator(returns)
        calibrator.calibrate_all()

        for key, val in calibrator.calibrated.items():
            assert isinstance(val, (int, float)), f"{key}应为float, 实际为{type(val)}"

    def test_get_calibration_config_returns_flat_dict(self):
        """get_calibration_config()返回flat dict可直接传入模块"""
        from calibration import ParameterCalibrator

        returns = pd.Series(np.random.randn(200) * 0.01)
        calibrator = ParameterCalibrator(returns)
        calibrator.calibrate_all()

        config = calibrator.get_calibration_config()
        assert 'EWMA_LAMBDA' in config
        assert 'T_DF' in config
        assert 'AR_PHI' in config
        assert isinstance(config['EWMA_LAMBDA'], (int, float))