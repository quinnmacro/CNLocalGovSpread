"""
content.py 测试 - 纯辅助函数

覆盖3个纯辅助函数 (无Streamlit依赖):
- get_spread_position_comment
- get_volatility_comment
- get_var_comment

Streamlit渲染函数不在测试范围内 (需要st运行环境)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

from src.content import (
    get_spread_position_comment,
    get_volatility_comment,
    get_var_comment,
)


# ============================================================================
# get_spread_position_comment
# ============================================================================

class TestGetSpreadPositionComment:

    def test_high_position_z2_plus(self):
        # z_score > 2 requires (current-mean)/std > 2 strictly
        result = get_spread_position_comment(105, 80, 10)
        assert '历史高位' in result
        assert '均值回归' in result

    def test_above_mean_z1_to_z2(self):
        # z_score > 1 requires (current-mean)/std > 1 strictly
        result = get_spread_position_comment(95, 80, 10)
        assert '偏高区间' in result

    def test_near_mean_z_minus1_to_z1(self):
        result = get_spread_position_comment(85, 80, 10)
        assert '正常区间' in result

    def test_below_mean_z_minus2_to_minus1(self):
        result = get_spread_position_comment(70, 80, 10)
        assert '偏低区间' in result

    def test_low_position_z_minus2(self):
        result = get_spread_position_comment(55, 80, 10)
        assert '历史低位' in result
        assert '反弹' in result

    def test_zero_std_returns_normal(self):
        result = get_spread_position_comment(100, 80, 0)
        assert '正常区间' in result

    def test_exact_mean(self):
        result = get_spread_position_comment(80, 80, 10)
        assert '正常区间' in result

    def test_z_score_format(self):
        result = get_spread_position_comment(100, 80, 10)
        assert 'σ' in result

    def test_bold_markdown_format(self):
        result = get_spread_position_comment(100, 80, 10)
        assert '**' in result


# ============================================================================
# get_volatility_comment
# ============================================================================

class TestGetVolatilityComment:

    def test_high_volatility_ratio_above_1_5(self):
        # ratio > 1.5 requires current/mean > 1.5 strictly
        result = get_volatility_comment(0.031, 0.02)
        assert '高波动状态' in result
        assert '风险敞口' in result

    def test_slightly_high_ratio_1_2_to_1_5(self):
        # ratio > 1.2 requires current/mean > 1.2 strictly
        result = get_volatility_comment(0.026, 0.02)
        assert '偏高' in result

    def test_normal_volatility_ratio_0_8_to_1_2(self):
        result = get_volatility_comment(0.02, 0.02)
        assert '正常' in result

    def test_low_volatility_ratio_below_0_8(self):
        result = get_volatility_comment(0.01, 0.02)
        assert '低波动' in result
        assert '方向性交易' in result

    def test_zero_mean_vol_returns_normal(self):
        result = get_volatility_comment(0.02, 0)
        assert '正常' in result

    def test_exact_mean_vol(self):
        result = get_volatility_comment(0.02, 0.02)
        assert '1.0倍' in result

    def test_ratio_format(self):
        result = get_volatility_comment(0.03, 0.02)
        assert '倍' in result

    def test_bold_markdown(self):
        result = get_volatility_comment(0.03, 0.02)
        assert '**' in result


# ============================================================================
# get_var_comment
# ============================================================================

class TestGetVarComment:

    def test_high_tail_ratio_above_1_5(self):
        result = get_var_comment(-0.05, -0.08)
        assert '尾部风险较高' in result
        assert '远超VaR' in result

    def test_moderate_tail_ratio_1_2_to_1_5(self):
        result = get_var_comment(-0.05, -0.065)
        assert '厚尾特征' in result

    def test_low_tail_ratio_below_1_2(self):
        result = get_var_comment(-0.05, -0.055)
        assert '可控' in result

    def test_positive_var_and_es(self):
        # VaR and ES can be positive in some contexts
        result = get_var_comment(0.05, 0.08)
        assert '比率' in result

    def test_none_var_uses_default(self):
        result = get_var_comment(None, -0.08)
        assert '比率' in result

    def test_none_es_uses_default(self):
        result = get_var_comment(-0.05, None)
        assert '比率' in result

    def test_both_none(self):
        result = get_var_comment(None, None)
        assert '比率' in result

    def test_zero_var_with_nonzero_es(self):
        result = get_var_comment(0, -0.08)
        assert '比率' in result

    def test_ratio_format_decimal(self):
        result = get_var_comment(-0.05, -0.08)
        assert '1.60' in result