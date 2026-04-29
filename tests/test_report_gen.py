"""
report_gen.py 测试 - 报告生成模块

覆盖 ReportGenerator 类和3个便捷函数:
- 初始化/历史加载/保存
- generate_report (PDF/Excel/HTML/Text 格式)
- _prepare_report_data (各章节数据准备)
- _generate_recommendation (交易建议生成)
- _add_to_history / get_history / delete_report
- generate_report / get_report_history / generate_quick_report 便捷函数
- DISCLAIMER 常量
"""

import pytest
import os
import json
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.report_gen import (
    ReportGenerator,
    DISCLAIMER,
    generate_report,
    get_report_history,
    generate_quick_report,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tmp_output_dir(tmp_path):
    """临时报告输出目录"""
    return str(tmp_path / 'reports')


@pytest.fixture
def clean_data():
    """模拟清洗后的数据"""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    np.random.seed(42)
    spread = 80 + np.cumsum(np.random.randn(100) * 0.3)
    return pd.DataFrame({'spread': spread}, index=dates)


@pytest.fixture
def returns():
    """模拟收益率序列"""
    dates = pd.date_range('2020-01-02', periods=99, freq='D')
    np.random.seed(42)
    return pd.Series(np.random.randn(99) * 0.02, index=dates, name='returns')


@pytest.fixture
def mock_kalman():
    """模拟卡尔曼滤波器"""
    kalman = MagicMock()
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    kalman.smoothed_state = pd.Series(80 + np.random.randn(100) * 0.5, index=dates)
    kalman.get_signal_deviation.return_value = pd.Series(np.random.randn(100), index=dates)
    return kalman


@pytest.fixture
def mock_vol_modeler():
    """模拟波动率建模器"""
    vol = MagicMock()
    vol.run_tournament.return_value = 'GARCH'
    vol.ic_scores = {'GARCH': {'AIC': -500.0, 'BIC': -480.0}}
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    vol.get_conditional_volatility.return_value = pd.Series(
        np.abs(np.random.randn(100)) * 0.02 + 0.01, index=dates
    )
    return vol


@pytest.fixture
def mock_evt():
    """模拟EVT分析器"""
    evt = MagicMock()
    evt.var = -0.05
    evt.es = -0.08
    evt.calculate_var.return_value = -0.05
    evt.calculate_es.return_value = -0.08
    evt.get_tail_index.return_value = 0.3
    evt.gpd_params = {'shape': 0.15, 'scale': 0.02}
    return evt


@pytest.fixture
def generator(tmp_output_dir):
    """创建报告生成器实例"""
    return ReportGenerator(output_dir=tmp_output_dir)


# ============================================================================
# DISCLAIMER
# ============================================================================

class TestDisclaimer:

    def test_disclaimer_exists(self):
        assert DISCLAIMER is not None
        assert len(DISCLAIMER) > 0

    def test_disclaimer_contains_version(self):
        assert '3.0.0' in DISCLAIMER

    def test_disclaimer_contains_warning(self):
        assert '不构成任何投资建议' in DISCLAIMER

    def test_disclaimer_contains_risk_notice(self):
        assert '投资有风险' in DISCLAIMER


# ============================================================================
# ReportGenerator.__init__
# ============================================================================

class TestReportGeneratorInit:

    def test_creates_output_dir(self, tmp_output_dir):
        gen = ReportGenerator(output_dir=tmp_output_dir)
        assert os.path.isdir(tmp_output_dir)

    def test_default_output_dir(self):
        gen = ReportGenerator()
        assert gen.output_dir == 'reports'

    def test_history_file_path(self, tmp_output_dir):
        gen = ReportGenerator(output_dir=tmp_output_dir)
        expected = os.path.join(tmp_output_dir, 'history.json')
        assert gen.report_history_file == expected

    def test_empty_history_on_new_dir(self, tmp_output_dir):
        gen = ReportGenerator(output_dir=tmp_output_dir)
        assert gen.history == []


# ============================================================================
# _load_history / _save_history
# ============================================================================

class TestHistoryManagement:

    def test_load_history_from_file(self, tmp_output_dir):
        os.makedirs(tmp_output_dir, exist_ok=True)
        history = [{'title': 'test', 'format': 'PDF', 'path': '/tmp/test.pdf'}]
        with open(os.path.join(tmp_output_dir, 'history.json'), 'w') as f:
            json.dump(history, f)

        gen = ReportGenerator(output_dir=tmp_output_dir)
        assert len(gen.history) == 1
        assert gen.history[0]['title'] == 'test'

    def test_save_history_creates_file(self, tmp_output_dir):
        gen = ReportGenerator(output_dir=tmp_output_dir)
        gen.history = [{'title': 'saved', 'format': 'HTML'}]
        gen._save_history()

        with open(os.path.join(tmp_output_dir, 'history.json'), 'r') as f:
            loaded = json.load(f)
        assert loaded[0]['title'] == 'saved'

    def test_get_history_returns_list(self, generator):
        result = generator.get_history()
        assert isinstance(result, list)


# ============================================================================
# _prepare_report_data
# ============================================================================

class TestPrepareReportData:

    def test_overview_section(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        data = generator._prepare_report_data(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            sections=['数据概览']
        )
        assert 'overview' in data
        assert '当前利差' in data['overview']
        assert '历史均值' in data['overview']

    def test_signal_section(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        data = generator._prepare_report_data(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            sections=['信号分析']
        )
        assert 'signal' in data
        assert '交易信号' in data['signal']

    def test_signal_without_kalman(self, generator, clean_data, returns, mock_vol_modeler, mock_evt):
        data = generator._prepare_report_data(
            clean_data, returns, None, mock_vol_modeler, mock_evt,
            sections=['信号分析']
        )
        assert 'signal' not in data

    def test_volatility_section(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        data = generator._prepare_report_data(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            sections=['波动率分析']
        )
        assert 'volatility' in data
        assert '获胜模型' in data['volatility']

    def test_risk_section(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        data = generator._prepare_report_data(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            sections=['风险分析']
        )
        assert 'risk' in data
        assert '99% VaR' in data['risk']

    def test_recommendation_section(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        data = generator._prepare_report_data(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            sections=['交易建议']
        )
        assert 'recommendation' in data
        assert '建议' in data['recommendation']

    def test_all_sections(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        data = generator._prepare_report_data(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            sections=['数据概览', '信号分析', '波动率分析', '风险分析', '交易建议']
        )
        assert len(data) >= 4


# ============================================================================
# _generate_recommendation
# ============================================================================

class TestGenerateRecommendation:

    def test_with_kalman_high_deviation(self, generator, clean_data, mock_kalman):
        # Set deviation > 1.5 (高估 → 做空)
        mock_kalman.get_signal_deviation.return_value = pd.Series([2.0])
        rec = generator._generate_recommendation(clean_data, mock_kalman, None, None)
        assert rec['建议'] == '做空'

    def test_with_kalman_low_deviation(self, generator, clean_data, mock_kalman):
        # Set deviation < -1.5 (低估 → 做多)
        mock_kalman.get_signal_deviation.return_value = pd.Series([-2.0])
        rec = generator._generate_recommendation(clean_data, mock_kalman, None, None)
        assert rec['建议'] == '做多'

    def test_with_kalman_normal_deviation(self, generator, clean_data, mock_kalman):
        mock_kalman.get_signal_deviation.return_value = pd.Series([0.5])
        rec = generator._generate_recommendation(clean_data, mock_kalman, None, None)
        assert rec['建议'] == '中性'

    def test_without_kalman(self, generator, clean_data):
        rec = generator._generate_recommendation(clean_data, None, None, None)
        assert '建议' in rec
        assert '当前利差' in rec

    def test_stop_loss_with_evt_var(self, generator, clean_data, mock_evt):
        mock_evt.var = -0.05
        rec = generator._generate_recommendation(clean_data, None, mock_evt, None)
        assert '止损建议' in rec

    def test_stop_loss_without_evt(self, generator, clean_data):
        rec = generator._generate_recommendation(clean_data, None, None, None)
        assert '止损建议' in rec


# ============================================================================
# generate_report (format dispatch)
# ============================================================================

class TestGenerateReportFormats:

    def test_html_format(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        path = generator.generate_report(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            format="HTML"
        )
        assert path.endswith('.html')
        assert os.path.exists(path)

    def test_html_contains_title(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        path = generator.generate_report(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            title="测试报告", format="HTML"
        )
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '测试报告' in content

    def test_html_contains_disclaimer(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        path = generator.generate_report(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            format="HTML"
        )
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '重要声明' in content

    def test_html_contains_version(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        path = generator.generate_report(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            format="HTML"
        )
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'v3.0.0' in content

    def test_excel_format(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        path = generator.generate_report(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            format="Excel"
        )
        assert path.endswith('.xlsx')
        assert os.path.exists(path)

    def test_text_format_fallback(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        # TXT is not a supported format - should raise ValueError
        with pytest.raises(ValueError, match='不支持的格式'):
            generator.generate_report(
                clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
                format="TXT"
            )

    def test_unsupported_format_raises(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        with pytest.raises(ValueError, match='不支持的格式'):
            generator.generate_report(
                clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
                format="DOCX"
            )

    def test_custom_sections(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        path = generator.generate_report(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            format="HTML", sections=['数据概览']
        )
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'overview' in content

    def test_default_sections(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        path = generator.generate_report(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            format="HTML"
        )
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'overview' in content

    def test_pdf_format_or_text_fallback(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        path = generator.generate_report(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            format="PDF"
        )
        # Either PDF or text fallback depending on reportlab availability
        assert os.path.exists(path)


# ============================================================================
# _generate_html internal
# ============================================================================

class TestGenerateHtmlInternal:

    def test_html_structure(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        data = generator._prepare_report_data(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            sections=['数据概览']
        )
        path = generator._generate_html(data, "Test Report", "20200101_120000")
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '<!DOCTYPE html>' in content
        assert '</html>' in content

    def test_html_tables(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        data = generator._prepare_report_data(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            sections=['数据概览']
        )
        path = generator._generate_html(data, "Test", "20200101")
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '<table>' in content


# ============================================================================
# _generate_text internal
# ============================================================================

class TestGenerateTextInternal:

    def test_text_format(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        data = generator._prepare_report_data(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            sections=['数据概览']
        )
        path = generator._generate_text(data, "Test Report", "20200101_120000", 'txt')
        assert path.endswith('.txt')
        assert os.path.exists(path)

    def test_text_content(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        data = generator._prepare_report_data(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            sections=['数据概览']
        )
        path = generator._generate_text(data, "Test Report", "20200101", 'txt')
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert 'Test Report' in content
        assert 'overview' in content


# ============================================================================
# _add_to_history / get_history / delete_report
# ============================================================================

class TestHistoryOps:

    def test_add_to_history(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        generator.generate_report(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            format="HTML"
        )
        history = generator.get_history()
        assert len(history) == 1
        assert history[0]['format'] == 'HTML'

    def test_multiple_reports_in_history(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        generator.generate_report(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            format="HTML"
        )
        generator.generate_report(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            format="HTML"
        )
        history = generator.get_history()
        assert len(history) == 2

    def test_delete_report(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        path = generator.generate_report(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            format="HTML"
        )
        filename = os.path.basename(path)
        result = generator.delete_report(filename)
        assert result is True
        assert not os.path.exists(path)
        assert len(generator.get_history()) == 0

    def test_delete_nonexistent_report(self, generator):
        result = generator.delete_report('nonexistent.html')
        assert result is False

    def test_history_record_fields(self, generator, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        generator.generate_report(
            clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt,
            title="测试标题", format="HTML"
        )
        record = generator.get_history()[0]
        assert 'title' in record
        assert 'format' in record
        assert 'path' in record
        assert 'timestamp' in record
        assert 'filename' in record
        assert record['title'] == '测试标题'


# ============================================================================
# Convenience functions
# ============================================================================

class TestConvenienceFunctions:

    def test_generate_report_function(self, tmp_path, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        with patch.object(ReportGenerator, '__init__', return_value=None):
            with patch.object(ReportGenerator, 'generate_report', return_value='/tmp/test.html') as mock_gen:
                result = generate_report(clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt)
                assert mock_gen.called

    def test_get_report_history_function(self, tmp_output_dir):
        gen = ReportGenerator(output_dir=tmp_output_dir)
        # Empty history → None
        with patch('src.report_gen.ReportGenerator') as mock_class:
            mock_instance = MagicMock()
            mock_instance.get_history.return_value = []
            mock_class.return_value = mock_instance
            result = get_report_history()
            assert result is None

    def test_generate_quick_report_function(self, clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt):
        with patch.object(ReportGenerator, '__init__', return_value=None):
            with patch.object(ReportGenerator, 'generate_report', return_value='/tmp/quick.html') as mock_gen:
                result = generate_quick_report(clean_data, returns, mock_kalman, mock_vol_modeler, mock_evt)
                assert mock_gen.called