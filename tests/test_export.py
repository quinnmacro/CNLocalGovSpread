"""
export.py 测试 - 数据导出模块

覆盖 export_to_excel 函数:
- 基本导出功能
- 多Sheet结构
- 部分数据导出
- 风险指标Sheet
"""

import pytest
import os
import numpy as np
import pandas as pd
from datetime import datetime

from src.export import export_to_excel


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def dates():
    return pd.date_range('2020-01-01', periods=50, freq='D')


@pytest.fixture
def clean_data(dates):
    np.random.seed(42)
    return pd.DataFrame({'spread': 80 + np.random.randn(50) * 0.5}, index=dates)


@pytest.fixture
def returns(dates):
    np.random.seed(42)
    return pd.Series(np.random.randn(49) * 0.02, index=dates[1:], name='returns')


@pytest.fixture
def smoothed_spread(dates):
    np.random.seed(42)
    return pd.Series(80 + np.random.randn(50) * 0.3, index=dates, name='smoothed')


@pytest.fixture
def signal_deviation(dates):
    np.random.seed(42)
    return pd.Series(np.random.randn(50), index=dates, name='deviation')


@pytest.fixture
def winner_volatility(dates):
    np.random.seed(42)
    return pd.Series(np.abs(np.random.randn(50)) * 0.02 + 0.01, index=dates, name='vol')


@pytest.fixture
def output_path(tmp_path):
    return str(tmp_path / 'test_export.xlsx')


# ============================================================================
# Full export
# ============================================================================

class TestFullExport:

    def test_returns_output_path(self, output_path, clean_data, returns, smoothed_spread, signal_deviation, winner_volatility):
        path = export_to_excel(
            output_path, clean_data, returns, smoothed_spread,
            signal_deviation, winner_volatility, 'GARCH',
            evt_var=-5.0, evt_es=-8.0
        )
        assert path == output_path
        assert os.path.exists(output_path)

    def test_file_created(self, output_path, clean_data, returns, smoothed_spread, signal_deviation, winner_volatility):
        export_to_excel(
            output_path, clean_data, returns, smoothed_spread,
            signal_deviation, winner_volatility
        )
        assert os.path.exists(output_path)

    def test_all_sheets_present(self, output_path, clean_data, returns, smoothed_spread, signal_deviation, winner_volatility):
        export_to_excel(
            output_path, clean_data, returns, smoothed_spread,
            signal_deviation, winner_volatility, 'GARCH',
            evt_var=-5.0, evt_es=-8.0
        )
        xl = pd.ExcelFile(output_path)
        sheet_names = xl.sheet_names
        assert '原始数据' in sheet_names
        assert '利差变化' in sheet_names
        assert '信号分析' in sheet_names
        assert '波动率' in sheet_names
        assert '风险指标' in sheet_names


# ============================================================================
# Partial data export
# ============================================================================

class TestPartialExport:

    def test_only_clean_data(self, output_path, clean_data):
        path = export_to_excel(output_path, clean_data=clean_data)
        assert os.path.exists(path)
        xl = pd.ExcelFile(path)
        assert '原始数据' in xl.sheet_names

    def test_only_returns(self, output_path, returns):
        path = export_to_excel(output_path, returns=returns)
        assert os.path.exists(path)
        xl = pd.ExcelFile(path)
        assert '利差变化' in xl.sheet_names

    def test_no_data_just_risk(self, output_path):
        path = export_to_excel(output_path, evt_var=-5.0, evt_es=-8.0)
        assert os.path.exists(path)
        xl = pd.ExcelFile(path)
        assert '风险指标' in xl.sheet_names

    def test_signal_analysis_requires_both(self, output_path, clean_data, smoothed_spread):
        # Only smoothed_spread without signal_deviation → no signal sheet
        path = export_to_excel(output_path, clean_data=clean_data, smoothed_spread=smoothed_spread)
        xl = pd.ExcelFile(path)
        assert '信号分析' not in xl.sheet_names

    def test_signal_with_both(self, output_path, clean_data, smoothed_spread, signal_deviation):
        path = export_to_excel(
            output_path, clean_data=clean_data,
            smoothed_spread=smoothed_spread, signal_deviation=signal_deviation
        )
        xl = pd.ExcelFile(path)
        assert '信号分析' in xl.sheet_names


# ============================================================================
# Risk metrics sheet
# ============================================================================

class TestRiskMetricsSheet:

    def test_var_in_risk_sheet(self, output_path):
        path = export_to_excel(output_path, evt_var=-5.0)
        risk_df = pd.read_excel(path, sheet_name='风险指标')
        assert 'VaR_99%_(bps)' in risk_df.columns

    def test_es_in_risk_sheet(self, output_path):
        path = export_to_excel(output_path, evt_es=-8.0)
        risk_df = pd.read_excel(path, sheet_name='风险指标')
        assert 'ES_99%_(bps)' in risk_df.columns

    def test_winner_model_in_risk_sheet(self, output_path):
        path = export_to_excel(output_path, winner_model='EGARCH')
        risk_df = pd.read_excel(path, sheet_name='风险指标')
        assert 'Winner_Model' in risk_df.columns
        assert risk_df['Winner_Model'].iloc[0] == 'EGARCH'

    def test_config_in_risk_sheet(self, output_path):
        config = {'SOURCE': 'Wind', 'START_DATE': '2020-01', 'END_DATE': '2023-12'}
        path = export_to_excel(output_path, config=config)
        risk_df = pd.read_excel(path, sheet_name='风险指标')
        assert 'Data_Source' in risk_df.columns

    def test_export_time_present(self, output_path):
        path = export_to_excel(output_path)
        risk_df = pd.read_excel(path, sheet_name='风险指标')
        assert 'Export_Time' in risk_df.columns

    def test_var_rounding(self, output_path):
        path = export_to_excel(output_path, evt_var=-5.123)
        risk_df = pd.read_excel(path, sheet_name='风险指标')
        var_val = risk_df['VaR_99%_(bps)'].iloc[0]
        assert abs(var_val - (-5.12)) < 0.01


# ============================================================================
# Volatility sheet
# ============================================================================

class TestVolatilitySheet:

    def test_volatility_with_model_name(self, output_path, winner_volatility):
        path = export_to_excel(output_path, winner_volatility=winner_volatility, winner_model='GARCH')
        vol_df = pd.read_excel(path, sheet_name='波动率', index_col=0)
        assert 'GARCH_volatility' in vol_df.columns

    def test_volatility_without_model_name(self, output_path, winner_volatility):
        path = export_to_excel(output_path, winner_volatility=winner_volatility)
        vol_df = pd.read_excel(path, sheet_name='波动率', index_col=0)
        assert 'conditional_volatility' in vol_df.columns