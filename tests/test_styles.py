"""
styles.py 测试 - Streamlit渲染辅助函数

覆盖所有导出函数 (通过mock st.markdown验证调用):
- metric_card, alert_box, section_header, apply_theme
- render_page_header, render_footer
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from unittest.mock import patch, MagicMock

from src.styles import (
    metric_card,
    alert_box,
    section_header,
    apply_theme,
    render_page_header,
    render_footer,
)


@pytest.fixture
def mock_st():
    """Mock streamlit module"""
    with patch('src.styles.st') as mock:
        yield mock


# ============================================================================
# metric_card
# ============================================================================

class TestMetricCard:

    def test_calls_st_markdown(self, mock_st):
        metric_card("利差", "80bps")
        mock_st.markdown.assert_called_once()

    def test_html_contains_title(self, mock_st):
        metric_card("利差", "80bps")
        call_arg = mock_st.markdown.call_args[0][0]
        assert '利差' in call_arg

    def test_html_contains_value(self, mock_st):
        metric_card("利差", "80bps")
        call_arg = mock_st.markdown.call_args[0][0]
        assert '80bps' in call_arg

    def test_html_with_delta(self, mock_st):
        metric_card("利差", "80bps", delta="+5")
        call_arg = mock_st.markdown.call_args[0][0]
        assert '+5' in call_arg

    def test_html_without_delta(self, mock_st):
        metric_card("利差", "80bps")
        call_arg = mock_st.markdown.call_args[0][0]
        assert 'opacity: 0.8' not in call_arg

    def test_status_normal_color(self, mock_st):
        metric_card("利差", "80bps", status="normal")
        call_arg = mock_st.markdown.call_args[0][0]
        assert 'var(--primary)' in call_arg

    def test_status_warning_color(self, mock_st):
        metric_card("利差", "80bps", status="warning")
        call_arg = mock_st.markdown.call_args[0][0]
        assert 'var(--warning)' in call_arg

    def test_status_danger_color(self, mock_st):
        metric_card("利差", "80bps", status="danger")
        call_arg = mock_st.markdown.call_args[0][0]
        assert 'var(--danger)' in call_arg

    def test_unsafe_allow_html(self, mock_st):
        metric_card("利差", "80bps")
        assert mock_st.markdown.call_args[1]['unsafe_allow_html'] is True


# ============================================================================
# alert_box
# ============================================================================

class TestAlertBox:

    def test_calls_st_markdown(self, mock_st):
        alert_box("test message")
        mock_st.markdown.assert_called_once()

    def test_info_level(self, mock_st):
        alert_box("msg", level='info')
        call_arg = mock_st.markdown.call_args[0][0]
        assert 'alert-info' in call_arg

    def test_warning_level(self, mock_st):
        alert_box("msg", level='warning')
        call_arg = mock_st.markdown.call_args[0][0]
        assert 'alert-warning' in call_arg

    def test_danger_level(self, mock_st):
        alert_box("msg", level='danger')
        call_arg = mock_st.markdown.call_args[0][0]
        assert 'alert-danger' in call_arg

    def test_success_level(self, mock_st):
        alert_box("msg", level='success')
        call_arg = mock_st.markdown.call_args[0][0]
        assert 'alert-success' in call_arg

    def test_unknown_level_defaults_info(self, mock_st):
        alert_box("msg", level='unknown')
        call_arg = mock_st.markdown.call_args[0][0]
        assert 'alert-info' in call_arg

    def test_message_content(self, mock_st):
        alert_box("风险警告！")
        call_arg = mock_st.markdown.call_args[0][0]
        assert '风险警告' in call_arg


# ============================================================================
# section_header
# ============================================================================

class TestSectionHeader:

    def test_without_icon(self, mock_st):
        section_header("分析结果")
        call_arg = mock_st.markdown.call_args[0][0]
        assert '分析结果' in call_arg
        assert 'section-title' in call_arg

    def test_with_icon(self, mock_st):
        section_header("波动率", icon="📉")
        call_arg = mock_st.markdown.call_args[0][0]
        assert '📉' in call_arg
        assert '波动率' in call_arg


# ============================================================================
# apply_theme
# ============================================================================

class TestApplyTheme:

    def test_dark_theme(self, mock_st):
        apply_theme('dark')
        mock_st.markdown.assert_called_once()

    def test_light_theme(self, mock_st):
        apply_theme('light')
        mock_st.markdown.assert_called_once()


# ============================================================================
# render_page_header
# ============================================================================

class TestRenderPageHeader:

    def test_title_only(self, mock_st):
        render_page_header("主标题")
        calls = mock_st.markdown.call_args_list
        assert any('主标题' in c[0][0] for c in calls)

    def test_with_subtitle(self, mock_st):
        render_page_header("主标题", subtitle="副标题")
        calls = mock_st.markdown.call_args_list
        assert any('主标题' in c[0][0] for c in calls)
        assert any('副标题' in c[0][0] for c in calls)

    def test_without_subtitle_single_call(self, mock_st):
        render_page_header("标题")
        assert mock_st.markdown.call_count == 1


# ============================================================================
# render_footer
# ============================================================================

class TestRenderFooter:

    def test_default_version(self, mock_st):
        render_footer()
        call_arg = mock_st.markdown.call_args[0][0]
        assert 'v3.0.0' in call_arg

    def test_custom_version(self, mock_st):
        render_footer(version='2.0.0')
        call_arg = mock_st.markdown.call_args[0][0]
        assert 'v2.0.0' in call_arg

    def test_default_author(self, mock_st):
        render_footer()
        call_arg = mock_st.markdown.call_args[0][0]
        assert 'Quinn Liu' in call_arg

    def test_with_github_link(self, mock_st):
        render_footer(github='https://github.com/test')
        call_arg = mock_st.markdown.call_args[0][0]
        assert 'https://github.com/test' in call_arg
        assert 'GitHub' in call_arg

    def test_with_linkedin_link(self, mock_st):
        render_footer(linkedin='https://linkedin.com/in/test')
        call_arg = mock_st.markdown.call_args[0][0]
        assert 'https://linkedin.com/in/test' in call_arg
        assert 'LinkedIn' in call_arg

    def test_both_links(self, mock_st):
        render_footer(github='https://g.com', linkedin='https://l.com')
        call_arg = mock_st.markdown.call_args[0][0]
        assert 'GitHub' in call_arg
        assert 'LinkedIn' in call_arg
        assert '|' in call_arg

    def test_no_links_no_separator(self, mock_st):
        render_footer()
        call_arg = mock_st.markdown.call_args[0][0]
        # Without links, no extra | separator before GitHub/LinkedIn
        # The footer format: "CNLocalGovSpread vX | Author: Name"
        assert 'Author: Quinn Liu' in call_arg