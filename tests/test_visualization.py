"""
visualization.py 测试 - 交互式图表模块

覆盖所有10个导出函数:
- add_range_selector, get_theme_config
- plot_signal_trend, plot_volatility_structure, plot_tail_risk
- print_var_comparison
- plot_multi_tenor_spread, plot_tenor_spread_correlation, plot_tenor_spread_statistics
- plot_credit_spread_comparison, plot_spread_premium_analysis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.visualization import (
    add_range_selector,
    get_theme_config,
    plot_signal_trend,
    plot_volatility_structure,
    plot_tail_risk,
    print_var_comparison,
    plot_multi_tenor_spread,
    plot_tenor_spread_correlation,
    plot_tenor_spread_statistics,
    plot_credit_spread_comparison,
    plot_spread_premium_analysis,
)


# ============================================================================
# Fixtures: mock data
# ============================================================================

@pytest.fixture
def dates():
    """生成日期索引"""
    return pd.date_range('2020-01-01', periods=200, freq='D')


@pytest.fixture
def clean_data(dates):
    """模拟清洗后的数据"""
    np.random.seed(42)
    spread = 80 + np.cumsum(np.random.randn(200) * 0.5)
    return pd.DataFrame({'spread': spread}, index=dates)


@pytest.fixture
def smoothed_spread(dates):
    """模拟卡尔曼平滑利差"""
    np.random.seed(42)
    values = 80 + np.cumsum(np.random.randn(200) * 0.3)
    return pd.Series(values, index=dates, name='smoothed')


@pytest.fixture
def signal_deviation(dates):
    """模拟信号偏离度"""
    np.random.seed(42)
    values = np.random.randn(200)
    # 确保有一些超出±1.5σ的点
    values[10] = -2.5
    values[50] = 2.5
    return pd.Series(values, index=dates, name='deviation')


@pytest.fixture
def returns(dates):
    """模拟收益率"""
    np.random.seed(42)
    values = np.random.randn(199) * 0.02  # 比clean_data少一行(diff)
    return pd.Series(values, index=dates[1:], name='returns')


@pytest.fixture
def winner_volatility(dates):
    """模拟条件波动率"""
    np.random.seed(42)
    values = np.abs(np.random.randn(200)) * 0.02 + 0.01
    # 插入一些高波动期
    values[180:] = values[180:] * 3
    return pd.Series(values, index=dates, name='volatility')


@pytest.fixture
def multi_tenor_df(dates):
    """模拟多期限利差数据"""
    np.random.seed(42)
    return pd.DataFrame({
        'spread_all': 80 + np.cumsum(np.random.randn(200) * 0.3),
        'spread_5y': 60 + np.cumsum(np.random.randn(200) * 0.2),
        'spread_10y': 90 + np.cumsum(np.random.randn(200) * 0.4),
        'spread_30y': 120 + np.cumsum(np.random.randn(200) * 0.5),
    }, index=dates)


@pytest.fixture
def local_gov_df(dates):
    """模拟地方债利差数据"""
    np.random.seed(42)
    return pd.DataFrame({
        'spread_all': 80 + np.cumsum(np.random.randn(200) * 0.3),
    }, index=dates)


@pytest.fixture
def credit_df(dates):
    """模拟信用利差数据"""
    np.random.seed(42)
    return pd.DataFrame({
        'credit_corp_aaa': 90 + np.cumsum(np.random.randn(200) * 0.25),
        'credit_corp_aa': 110 + np.cumsum(np.random.randn(200) * 0.35),
    }, index=dates)


# ============================================================================
# add_range_selector
# ============================================================================

class TestAddRangeSelector:

    def test_returns_figure(self):
        fig = go.Figure()
        result = add_range_selector(fig, dark_mode=False)
        assert isinstance(result, go.Figure)

    def test_light_mode_colors(self):
        fig = go.Figure()
        result = add_range_selector(fig, dark_mode=False)
        selector = result.layout.xaxis.rangeselector
        assert selector.font.color == '#1E3A5F'
        assert selector.bgcolor == 'rgba(255,255,255,0.8)'

    def test_dark_mode_colors(self):
        fig = go.Figure()
        result = add_range_selector(fig, dark_mode=True)
        selector = result.layout.xaxis.rangeselector
        assert selector.font.color == '#FAFAFA'
        assert selector.bgcolor == 'rgba(30,30,30,0.8)'

    def test_has_six_buttons(self):
        fig = go.Figure()
        result = add_range_selector(fig)
        buttons = result.layout.xaxis.rangeselector.buttons
        assert len(buttons) == 6

    def test_button_labels(self):
        fig = go.Figure()
        result = add_range_selector(fig)
        labels = [b.label for b in result.layout.xaxis.rangeselector.buttons]
        assert '1月' in labels
        assert '3月' in labels
        assert '全部' in labels

    def test_rangeslider_visible(self):
        fig = go.Figure()
        result = add_range_selector(fig)
        assert result.layout.xaxis.rangeslider.visible is True


# ============================================================================
# get_theme_config
# ============================================================================

class TestGetThemeConfig:

    def test_light_theme_keys(self):
        config = get_theme_config('light')
        expected_keys = ['template', 'paper_bgcolor', 'plot_bgcolor',
                         'font_color', 'grid_color', 'line_color', 'colors']
        assert all(k in config for k in expected_keys)

    def test_dark_theme_keys(self):
        config = get_theme_config('dark')
        expected_keys = ['template', 'paper_bgcolor', 'plot_bgcolor',
                         'font_color', 'grid_color', 'line_color', 'colors']
        assert all(k in config for k in expected_keys)

    def test_light_template(self):
        config = get_theme_config('light')
        assert config['template'] == 'none'

    def test_dark_template(self):
        config = get_theme_config('dark')
        assert config['template'] == 'plotly_dark'

    def test_light_font_color(self):
        config = get_theme_config('light')
        assert config['font_color'] == '#0F172A'

    def test_dark_font_color(self):
        config = get_theme_config('dark')
        assert config['font_color'] == '#F8FAFC'

    def test_colors_list_length(self):
        for theme in ['light', 'dark']:
            config = get_theme_config(theme)
            assert len(config['colors']) == 6

    def test_default_is_light(self):
        config = get_theme_config()
        assert config['template'] == 'none'


# ============================================================================
# plot_signal_trend
# ============================================================================

class TestPlotSignalTrend:

    def test_returns_figure(self, clean_data, smoothed_spread, signal_deviation):
        fig = plot_signal_trend(clean_data, smoothed_spread, signal_deviation)
        assert isinstance(fig, go.Figure)

    def test_has_raw_spread_trace(self, clean_data, smoothed_spread, signal_deviation):
        fig = plot_signal_trend(clean_data, smoothed_spread, signal_deviation)
        names = [t.name for t in fig.data]
        assert '原始利差' in names

    def test_has_kalman_trace(self, clean_data, smoothed_spread, signal_deviation):
        fig = plot_signal_trend(clean_data, smoothed_spread, signal_deviation)
        names = [t.name for t in fig.data]
        assert '卡尔曼趋势' in names

    def test_has_confidence_band(self, clean_data, smoothed_spread, signal_deviation):
        fig = plot_signal_trend(clean_data, smoothed_spread, signal_deviation)
        names = [t.name for t in fig.data]
        assert '参考区间 (±1.5σ, 残差)' in names

    def test_buy_signals_with_negative_deviation(self, clean_data, smoothed_spread, signal_deviation):
        fig = plot_signal_trend(clean_data, smoothed_spread, signal_deviation)
        names = [t.name for t in fig.data]
        assert '买入信号 (低估)' in names

    def test_sell_signals_with_positive_deviation(self, clean_data, smoothed_spread, signal_deviation):
        fig = plot_signal_trend(clean_data, smoothed_spread, signal_deviation)
        names = [t.name for t in fig.data]
        assert '卖出信号 (高估)' in names

    def test_dark_theme(self, clean_data, smoothed_spread, signal_deviation):
        fig = plot_signal_trend(clean_data, smoothed_spread, signal_deviation, theme='dark')
        assert fig.layout.template is not None
        assert fig.layout.paper_bgcolor == 'rgba(0,0,0,0)'

    def test_no_buy_signals_when_all_positive(self, clean_data, smoothed_spread, dates):
        all_positive = pd.Series(np.abs(np.random.randn(200)) + 2, index=dates)
        fig = plot_signal_trend(clean_data, smoothed_spread, all_positive)
        names = [t.name for t in fig.data]
        assert '买入信号 (低估)' not in names

    def test_no_sell_signals_when_all_negative(self, clean_data, smoothed_spread, dates):
        all_negative = pd.Series(-np.abs(np.random.randn(200)) - 2, index=dates)
        fig = plot_signal_trend(clean_data, smoothed_spread, all_negative)
        names = [t.name for t in fig.data]
        assert '卖出信号 (高估)' not in names

    def test_title_present(self, clean_data, smoothed_spread, signal_deviation):
        fig = plot_signal_trend(clean_data, smoothed_spread, signal_deviation)
        assert '信号提取' in fig.layout.title.text


# ============================================================================
# plot_volatility_structure
# ============================================================================

class TestPlotVolatilityStructure:

    def test_returns_figure(self, winner_volatility):
        fig = plot_volatility_structure(winner_volatility, 'GARCH')
        assert isinstance(fig, go.Figure)

    def test_has_volatility_trace(self, winner_volatility):
        fig = plot_volatility_structure(winner_volatility, 'EGARCH')
        names = [t.name for t in fig.data]
        assert 'EGARCH 条件波动率' in names

    def test_has_high_vol_markers(self, winner_volatility):
        fig = plot_volatility_structure(winner_volatility, 'GARCH')
        names = [t.name for t in fig.data]
        assert '高波动期 (危机模式)' in names

    def test_model_name_in_title(self, winner_volatility):
        fig = plot_volatility_structure(winner_volatility, 'GJR-GARCH')
        assert 'GJR-GARCH' in fig.layout.title.text

    def test_dark_theme(self, winner_volatility):
        fig = plot_volatility_structure(winner_volatility, 'GARCH', theme='dark')
        assert fig.layout.paper_bgcolor == 'rgba(0,0,0,0)'

    def test_threshold_hline_exists(self, winner_volatility):
        fig = plot_volatility_structure(winner_volatility, 'GARCH')
        # plotly stores hlines in layout.shapes
        assert len(fig.layout.shapes) > 0

    def test_no_high_vol_when_flat(self, dates):
        flat_vol = pd.Series(np.ones(200) * 0.01, index=dates)
        fig = plot_volatility_structure(flat_vol, 'GARCH')
        names = [t.name for t in fig.data]
        # 90th percentile of constant series = constant itself, so all points > threshold = none
        # Actually all equal means quantile(0.9) = 0.01, and no point > 0.01 since all = 0.01
        assert '高波动期 (危机模式)' not in names


# ============================================================================
# plot_tail_risk
# ============================================================================

class TestPlotTailRisk:

    def test_returns_tuple(self, returns):
        result = plot_tail_risk(returns, -0.05)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_figure_is_plotly(self, returns):
        fig, evt_var, empirical_var = plot_tail_risk(returns, -0.05)
        assert isinstance(fig, go.Figure)

    def test_evt_var_returned(self, returns):
        fig, evt_var, empirical_var = plot_tail_risk(returns, -0.05)
        assert evt_var == -0.05

    def test_empirical_var_returned(self, returns):
        fig, evt_var, empirical_var = plot_tail_risk(returns, -0.05)
        assert isinstance(empirical_var, (float, np.floating))

    def test_has_histogram_trace(self, returns):
        fig, _, _ = plot_tail_risk(returns, -0.05)
        names = [t.name for t in fig.data]
        assert '实际分布' in names

    def test_has_student_t_trace(self, returns):
        fig, _, _ = plot_tail_risk(returns, -0.05)
        names = [t.name for t in fig.data]
        # Name format: 'Student-t 拟合 (df=X.X)'
        student_t_names = [n for n in names if 'Student-t' in n]
        assert len(student_t_names) > 0

    def test_has_normal_trace(self, returns):
        fig, _, _ = plot_tail_risk(returns, -0.05)
        names = [t.name for t in fig.data]
        assert '正态分布对比' in names

    def test_custom_confidence(self, returns):
        fig, evt, emp = plot_tail_risk(returns, -0.03, confidence=0.95)
        # empirical_var should be 95th percentile, not 99th
        assert emp > evt  # 95th quantile > 99th EVT-VaR for negative VaR

    def test_dark_theme(self, returns):
        fig, _, _ = plot_tail_risk(returns, -0.05, theme='dark')
        assert fig.layout.template is not None


# ============================================================================
# print_var_comparison
# ============================================================================

class TestPrintVarComparison:

    def test_evt_higher_prints_warning(self, capsys):
        # evt_var > empirical_var triggers "风险更高" branch
        print_var_comparison(-3.0, -5.0)
        output = capsys.readouterr().out
        assert 'EVT 估计的风险更高' in output
        assert 'EVT-VaR' in output

    def test_evt_lower_prints_info(self, capsys):
        # evt_var <= empirical_var triggers "尾部风险可控" branch
        print_var_comparison(-5.0, -3.0)
        output = capsys.readouterr().out
        assert '尾部风险可控' in output

    def test_prints_difference(self, capsys):
        print_var_comparison(-5.0, -3.0)
        output = capsys.readouterr().out
        assert '差异' in output

    def test_values_formatted_as_bps(self, capsys):
        print_var_comparison(-5.0, -3.0)
        output = capsys.readouterr().out
        assert 'bps' in output


# ============================================================================
# plot_multi_tenor_spread
# ============================================================================

class TestPlotMultiTenorSpread:

    def test_returns_figure(self, multi_tenor_df):
        fig = plot_multi_tenor_spread(multi_tenor_df)
        assert isinstance(fig, go.Figure)

    def test_default_columns(self, multi_tenor_df):
        fig = plot_multi_tenor_spread(multi_tenor_df)
        names = [t.name for t in fig.data]
        assert '综合利差' in names
        assert '5年期' in names

    def test_custom_columns(self, multi_tenor_df):
        fig = plot_multi_tenor_spread(multi_tenor_df, columns=['spread_5y', 'spread_10y'])
        names = [t.name for t in fig.data]
        assert '5年期' in names
        assert '10年期' in names
        assert '综合利差' not in names

    def test_no_available_columns_raises(self, dates):
        df = pd.DataFrame({'other': np.random.randn(200)}, index=dates)
        with pytest.raises(ValueError, match='没有可用的利差列'):
            plot_multi_tenor_spread(df)

    def test_dark_theme(self, multi_tenor_df):
        fig = plot_multi_tenor_spread(multi_tenor_df, theme='dark')
        assert fig.layout.paper_bgcolor == 'rgba(0,0,0,0)'

    def test_trace_count_matches_columns(self, multi_tenor_df):
        fig = plot_multi_tenor_spread(multi_tenor_df)
        assert len(fig.data) == 4


# ============================================================================
# plot_tenor_spread_correlation
# ============================================================================

class TestPlotTenorSpreadCorrelation:

    def test_returns_figure(self, multi_tenor_df):
        fig = plot_tenor_spread_correlation(multi_tenor_df)
        assert isinstance(fig, go.Figure)

    def test_heatmap_trace(self, multi_tenor_df):
        fig = plot_tenor_spread_correlation(multi_tenor_df)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_correlation_values_in_range(self, multi_tenor_df):
        fig = plot_tenor_spread_correlation(multi_tenor_df)
        z_values = fig.data[0].z
        flat = np.array(z_values).flatten()
        assert flat.min() >= -1.01
        assert flat.max() <= 1.01

    def test_diagonal_near_one(self, multi_tenor_df):
        fig = plot_tenor_spread_correlation(multi_tenor_df)
        z_values = np.array(fig.data[0].z)
        for i in range(z_values.shape[0]):
            assert abs(z_values[i, i] - 1.0) < 0.01

    def test_less_than_two_columns_raises(self, dates):
        df = pd.DataFrame({'spread_all': np.random.randn(200)}, index=dates)
        with pytest.raises(ValueError, match='至少需要2列数据'):
            plot_tenor_spread_correlation(df)

    def test_custom_columns(self, multi_tenor_df):
        fig = plot_tenor_spread_correlation(multi_tenor_df, columns=['spread_5y', 'spread_10y'])
        display_names = list(fig.data[0].x)
        assert '5Y' in display_names
        assert '10Y' in display_names

    def test_dark_theme(self, multi_tenor_df):
        fig = plot_tenor_spread_correlation(multi_tenor_df, theme='dark')
        assert fig.layout.paper_bgcolor == 'rgba(0,0,0,0)'


# ============================================================================
# plot_tenor_spread_statistics
# ============================================================================

class TestPlotTenorSpreadStatistics:

    def test_returns_figure(self, multi_tenor_df):
        fig = plot_tenor_spread_statistics(multi_tenor_df)
        assert isinstance(fig, go.Figure)

    def test_box_traces(self, multi_tenor_df):
        fig = plot_tenor_spread_statistics(multi_tenor_df)
        for trace in fig.data:
            assert isinstance(trace, go.Box)

    def test_trace_count_matches_columns(self, multi_tenor_df):
        fig = plot_tenor_spread_statistics(multi_tenor_df)
        assert len(fig.data) == 4

    def test_custom_columns(self, multi_tenor_df):
        fig = plot_tenor_spread_statistics(multi_tenor_df, columns=['spread_5y'])
        assert len(fig.data) == 1

    def test_no_legend(self, multi_tenor_df):
        fig = plot_tenor_spread_statistics(multi_tenor_df)
        assert fig.layout.showlegend is False

    def test_dark_theme(self, multi_tenor_df):
        fig = plot_tenor_spread_statistics(multi_tenor_df, theme='dark')
        assert fig.layout.paper_bgcolor == 'rgba(0,0,0,0)'

    def test_empty_available_cols(self, dates):
        df = pd.DataFrame({'other': np.random.randn(200)}, index=dates)
        fig = plot_tenor_spread_statistics(df)
        assert len(fig.data) == 0


# ============================================================================
# plot_credit_spread_comparison
# ============================================================================

class TestPlotCreditSpreadComparison:

    def test_returns_figure(self, local_gov_df):
        fig = plot_credit_spread_comparison(local_gov_df)
        assert isinstance(fig, go.Figure)

    def test_local_gov_baseline_trace(self, local_gov_df):
        fig = plot_credit_spread_comparison(local_gov_df)
        names = [t.name for t in fig.data]
        assert '地方债 (基准)' in names

    def test_with_credit_data(self, local_gov_df, credit_df):
        fig = plot_credit_spread_comparison(local_gov_df, credit_df,
                                             credit_columns=['credit_corp_aaa'])
        names = [t.name for t in fig.data]
        assert '地方债 (基准)' in names
        assert 'Corp Aaa' in names

    def test_no_spread_all_column(self, dates):
        df = pd.DataFrame({'other': np.random.randn(200)}, index=dates)
        fig = plot_credit_spread_comparison(df)
        assert len(fig.data) == 0

    def test_credit_columns_not_in_df(self, local_gov_df, credit_df):
        fig = plot_credit_spread_comparison(local_gov_df, credit_df,
                                             credit_columns=['nonexistent'])
        names = [t.name for t in fig.data]
        assert '地方债 (基准)' in names
        assert len(fig.data) == 1

    def test_dark_theme(self, local_gov_df):
        fig = plot_credit_spread_comparison(local_gov_df, theme='dark')
        assert fig.layout.paper_bgcolor == 'rgba(0,0,0,0)'

    def test_title_text(self, local_gov_df):
        fig = plot_credit_spread_comparison(local_gov_df)
        assert '信用利差对比' in fig.layout.title.text


# ============================================================================
# plot_spread_premium_analysis
# ============================================================================

class TestPlotSpreadPremiumAnalysis:

    def test_returns_placeholder_when_no_credit(self, local_gov_df):
        fig = plot_spread_premium_analysis(local_gov_df)
        assert isinstance(fig, go.Figure)
        # Should have annotation text
        assert len(fig.layout.annotations) > 0

    def test_placeholder_annotation_text(self, local_gov_df):
        fig = plot_spread_premium_analysis(local_gov_df)
        text = fig.layout.annotations[0].text
        assert '未配置' in text

    def test_with_credit_data(self, local_gov_df, credit_df):
        fig = plot_spread_premium_analysis(local_gov_df, credit_df)
        names = [t.name for t in fig.data]
        assert '信用溢价' in names

    def test_premium_line_trace(self, local_gov_df, credit_df):
        fig = plot_spread_premium_analysis(local_gov_df, credit_df)
        # Should have hline for average premium
        assert len(fig.layout.shapes) > 0

    def test_average_premium_annotation(self, local_gov_df, credit_df):
        fig = plot_spread_premium_analysis(local_gov_df, credit_df)
        # plotly stores hline annotations in fig.layout.annotations
        annotations = fig.layout.annotations
        has_avg = any('平均溢价' in a.text for a in annotations)
        assert has_avg

    def test_custom_credit_column(self, local_gov_df, credit_df):
        fig = plot_spread_premium_analysis(local_gov_df, credit_df,
                                            credit_column='credit_corp_aa')
        names = [t.name for t in fig.data]
        assert '信用溢价' in names

    def test_invalid_credit_column_returns_placeholder(self, local_gov_df, credit_df):
        fig = plot_spread_premium_analysis(local_gov_df, credit_df,
                                            credit_column='nonexistent')
        assert len(fig.data) == 0  # placeholder figure

    def test_dark_theme_placeholder(self, local_gov_df):
        fig = plot_spread_premium_analysis(local_gov_df, theme='dark')
        assert fig.layout.paper_bgcolor == 'rgba(0,0,0,0)'

    def test_dark_theme_with_credit(self, local_gov_df, credit_df):
        fig = plot_spread_premium_analysis(local_gov_df, credit_df, theme='dark')
        assert fig.layout.paper_bgcolor == 'rgba(0,0,0,0)'