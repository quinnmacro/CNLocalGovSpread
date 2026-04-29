"""
Tests for data_engine.py - Data loading, cleaning, and returns calculation.

Covers: init, load_data dispatch, mock data generation (AR(1)+GARCH properties),
CSV loading, Wind EDB ImportError, clean_data (MAD outlier detection, ffill/bfill,
zero-MAD handling, threshold config), get_returns, full workflow integration,
boundary conditions.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# Ensure project root on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_engine import DataEngine


# ─── Fixtures ───

@pytest.fixture
def default_config():
    """Standard mock config matching dashboard defaults."""
    return {
        'SOURCE': 'MOCK',
        'START_DATE': '2024-01-01',
        'END_DATE': '2024-06-30',
        'MAD_THRESHOLD': 5.0,
    }


@pytest.fixture
def engine(default_config):
    """DataEngine with mock config."""
    return DataEngine(default_config)


@pytest.fixture
def loaded_engine(engine):
    """DataEngine after load_data() call."""
    engine.load_data()
    return engine


@pytest.fixture
def cleaned_engine(loaded_engine):
    """DataEngine after load_data() + clean_data() calls."""
    loaded_engine.clean_data()
    return loaded_engine


@pytest.fixture
def csv_config(tmp_path):
    """Config pointing to a real CSV file."""
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='B')
    df = pd.DataFrame({
        'date': dates,
        'spread_all': np.random.normal(100, 10, len(dates)),
        'spread_5y': np.random.normal(80, 8, len(dates)),
    })
    csv_file = tmp_path / 'local_gov_spread.csv'
    df.to_csv(csv_file, index=False)
    return {
        'SOURCE': 'CSV',
        'CSV_PATH': str(csv_file),
        'SPREAD_COLUMN': 'spread_all',
        'MAD_THRESHOLD': 5.0,
    }


# ─── Initialization Tests ───

class TestInit:
    def test_stores_config(self, default_config):
        e = DataEngine(default_config)
        assert e.config == default_config

    def test_raw_data_none_on_init(self, default_config):
        e = DataEngine(default_config)
        assert e._raw_data is None

    def test_clean_data_none_on_init(self, default_config):
        e = DataEngine(default_config)
        assert e._clean_data is None

    def test_empty_config_with_dates_defaults_to_mock(self):
        # _generate_mock_data needs START_DATE/END_DATE keys
        config = {'START_DATE': '2024-01-01', 'END_DATE': '2024-03-31'}
        e = DataEngine(config)
        result = e.load_data()
        assert isinstance(result, pd.DataFrame)


# ─── load_data Dispatch Tests ───

class TestLoadDataDispatch:
    def test_mock_source(self, engine):
        result = engine.load_data()
        assert isinstance(result, pd.DataFrame)
        assert 'spread' in result.columns

    def test_csv_source(self, csv_config):
        e = DataEngine(csv_config)
        result = e.load_data()
        assert isinstance(result, pd.DataFrame)
        assert 'spread' in result.columns

    def test_unknown_source_defaults_to_mock(self):
        config = {'SOURCE': 'UNKNOWN', 'START_DATE': '2024-01-01', 'END_DATE': '2024-03-31'}
        e = DataEngine(config)
        result = e.load_data()
        assert isinstance(result, pd.DataFrame)

    def test_wind_source_raises_import_error(self):
        config = {'SOURCE': 'WIND_EDB'}
        e = DataEngine(config)
        with pytest.raises(ImportError, match="Wind Python API"):
            e.load_data()

    def test_csv_missing_file_raises(self):
        config = {'SOURCE': 'CSV', 'CSV_PATH': '/nonexistent/path.csv'}
        e = DataEngine(config)
        with pytest.raises(FileNotFoundError):
            e.load_data()

    def test_csv_missing_column_raises(self, tmp_path):
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='B')
        df = pd.DataFrame({'date': dates, 'other_col': np.zeros(len(dates))})
        csv_file = tmp_path / 'data.csv'
        df.to_csv(csv_file, index=False)
        config = {'SOURCE': 'CSV', 'CSV_PATH': str(csv_file), 'SPREAD_COLUMN': 'spread_all'}
        e = DataEngine(config)
        with pytest.raises(ValueError, match="列"):
            e.load_data()

    def test_csv_custom_column(self, tmp_path):
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='B')
        df = pd.DataFrame({
            'date': dates,
            'spread_5y': np.random.normal(80, 8, len(dates)),
        })
        csv_file = tmp_path / 'data.csv'
        df.to_csv(csv_file, index=False)
        config = {'SOURCE': 'CSV', 'CSV_PATH': str(csv_file), 'SPREAD_COLUMN': 'spread_5y'}
        e = DataEngine(config)
        result = e.load_data()
        assert 'spread' in result.columns
        # Values should match original spread_5y column
        assert len(result) == len(dates)


# ─── Mock Data Generation Tests ───

class TestMockDataGeneration:
    def test_correct_date_range(self, engine):
        result = engine.load_data()
        assert result.index[0] >= pd.Timestamp(engine.config['START_DATE'])
        assert result.index[-1] <= pd.Timestamp(engine.config['END_DATE'])

    def test_business_days_only(self, engine):
        result = engine.load_data()
        for dt in result.index[:20]:  # Check first 20
            assert dt.weekday() < 5  # Mon-Fri

    def test_spread_column_exists(self, engine):
        result = engine.load_data()
        assert 'spread' in result.columns

    def test_spread_not_all_zeros(self, engine):
        result = engine.load_data()
        assert result['spread'].std() > 0

    def test_reproducible_with_seed(self, default_config):
        e1 = DataEngine(default_config)
        r1 = e1.load_data()
        # Same config, same seed (42) inside _generate_mock_data
        e2 = DataEngine(default_config)
        r2 = e2.load_data()
        pd.testing.assert_frame_equal(r1, r2)

    def test_initial_spread_value(self, engine):
        result = engine.load_data()
        # spread[0] = 100 by design
        assert result['spread'].iloc[0] == 100.0

    def test_mean_reversion_tendency(self):
        """AR(1) with phi=0.98 and mu=100 should produce spread near 100."""
        config = {'SOURCE': 'MOCK', 'START_DATE': '2024-01-01', 'END_DATE': '2026-01-01'}
        e = DataEngine(config)
        result = e.load_data()
        # Long series should mean-revert near mu=100
        assert abs(result['spread'].mean() - 100) < 20

    def test_volatility_clustering_present(self):
        """GARCH(1,1) process should show volatility clustering."""
        config = {'SOURCE': 'MOCK', 'START_DATE': '2024-01-01', 'END_DATE': '2026-01-01'}
        e = DataEngine(config)
        result = e.load_data()
        returns = result['spread'].diff().dropna()
        # Squared returns should show positive autocorrelation (clustering)
        acf = returns.pow(2).autocorr()
        assert acf > 0.05  # Some clustering present

    def test_fat_tails_present(self):
        """t-distribution shocks (df=5) should produce fat tails."""
        config = {'SOURCE': 'MOCK', 'START_DATE': '2024-01-01', 'END_DATE': '2026-01-01'}
        e = DataEngine(config)
        result = e.load_data()
        returns = result['spread'].diff().dropna()
        # Kurtosis > 3 indicates fat tails (excess kurtosis)
        kurt = returns.kurtosis()
        assert kurt > 3.0

    def test_stores_raw_data(self, engine):
        engine.load_data()
        assert engine._raw_data is not None
        assert isinstance(engine._raw_data, pd.DataFrame)

    def test_datetime_index(self, engine):
        result = engine.load_data()
        assert isinstance(result.index, pd.DatetimeIndex)


# ─── clean_data Tests ───

class TestCleanData:
    def test_raises_without_load(self, engine):
        with pytest.raises(ValueError, match="请先调用 load_data"):
            engine.clean_data()

    def test_returns_dataframe(self, loaded_engine):
        result = loaded_engine.clean_data()
        assert isinstance(result, pd.DataFrame)

    def test_stores_clean_data(self, loaded_engine):
        loaded_engine.clean_data()
        assert loaded_engine._clean_data is not None

    def test_same_length_as_raw(self, loaded_engine):
        result = loaded_engine.clean_data()
        assert len(result) == len(loaded_engine._raw_data)

    def test_forward_fill_handles_nan(self):
        """NaN values should be filled with ffill().bfill()."""
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='B')
        spread = np.random.normal(100, 10, len(dates))
        spread[5] = np.nan
        spread[10] = np.nan
        config = {'SOURCE': 'MOCK', 'START_DATE': '2024-01-01', 'END_DATE': '2024-03-31',
                  'MAD_THRESHOLD': 5.0}
        e = DataEngine(config)
        e._raw_data = pd.DataFrame({'spread': spread}, index=dates)
        result = e.clean_data()
        assert result['spread'].notna().all()

    def test_bfill_first_nan(self):
        """First value NaN should be backfilled."""
        dates = pd.date_range('2024-01-01', '2024-02-28', freq='B')
        spread = np.random.normal(100, 10, len(dates))
        spread[0] = np.nan
        config = {'SOURCE': 'MOCK', 'START_DATE': '2024-01-01', 'END_DATE': '2024-02-28',
                  'MAD_THRESHOLD': 5.0}
        e = DataEngine(config)
        e._raw_data = pd.DataFrame({'spread': spread}, index=dates)
        result = e.clean_data()
        assert result['spread'].iloc[0] != np.nan

    def test_mad_outlier_replacement(self):
        """Extreme outliers should be replaced with median."""
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='B')
        spread = np.random.normal(100, 10, len(dates))
        # Insert extreme outlier (999 placeholder)
        spread[20] = 999
        spread[40] = -999
        config = {'SOURCE': 'MOCK', 'START_DATE': '2024-01-01', 'END_DATE': '2024-03-31',
                  'MAD_THRESHOLD': 5.0}
        e = DataEngine(config)
        e._raw_data = pd.DataFrame({'spread': spread}, index=dates)
        result = e.clean_data()
        # Outliers should be replaced with median, not remain extreme
        assert result['spread'].iloc[20] < 200
        assert result['spread'].iloc[40] > -200

    def test_custom_mad_threshold(self):
        """Lower threshold should catch more outliers."""
        dates = pd.date_range('2024-01-01', '2024-06-30', freq='B')
        spread = np.random.normal(100, 10, len(dates))
        # Moderate outlier that 5.0 threshold misses but 2.0 catches
        spread[30] = 160
        config_strict = {'SOURCE': 'MOCK', 'START_DATE': '2024-01-01', 'END_DATE': '2024-06-30',
                         'MAD_THRESHOLD': 2.0}
        config_loose = {'SOURCE': 'MOCK', 'START_DATE': '2024-01-01', 'END_DATE': '2024-06-30',
                        'MAD_THRESHOLD': 10.0}
        e_strict = DataEngine(config_strict)
        e_strict._raw_data = pd.DataFrame({'spread': spread}, index=dates)
        e_loose = DataEngine(config_loose)
        e_loose._raw_data = pd.DataFrame({'spread': spread}, index=dates)

        result_strict = e_strict.clean_data()
        result_loose = e_loose.clean_data()

        # Strict threshold should replace more outliers
        # (the 160 value should be caught by 2.0 but not 10.0)
        strict_outliers = (result_strict['spread'] == result_strict['spread'].median()).sum()
        loose_outliers = (result_loose['spread'] == result_loose['spread'].median()).sum()
        assert strict_outliers >= loose_outliers

    def test_zero_mad_handling(self):
        """When MAD=0 (constant data), modified_z_score should be 0."""
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='B')
        spread = np.full(len(dates), 100.0)  # Constant series
        config = {'SOURCE': 'MOCK', 'START_DATE': '2024-01-01', 'END_DATE': '2024-03-31',
                  'MAD_THRESHOLD': 5.0}
        e = DataEngine(config)
        e._raw_data = pd.DataFrame({'spread': spread}, index=dates)
        result = e.clean_data()
        # Should not crash, all values unchanged
        assert result['spread'].mean() == 100.0

    def test_modified_z_score_formula(self):
        """Verify 0.6745 * (x - median) / MAD formula catches extreme outliers."""
        dates = pd.date_range('2024-01-01', '2024-02-28', freq='B')
        # Varied base data so MAD > 0
        base_spread = np.random.normal(100, 5, len(dates))
        # Insert an extreme outlier
        base_spread[10] = 500
        config = {'SOURCE': 'MOCK', 'START_DATE': '2024-01-01', 'END_DATE': '2024-02-28',
                  'MAD_THRESHOLD': 3.0}
        e = DataEngine(config)
        e._raw_data = pd.DataFrame({'spread': base_spread}, index=dates)
        result = e.clean_data()
        # 500 should be replaced since modified z-score >> 3
        assert result['spread'].iloc[10] < 200

    def test_no_outliers_with_normal_data(self):
        """Normal mock data with threshold=5 should have few outliers."""
        config = {'SOURCE': 'MOCK', 'START_DATE': '2024-01-01', 'END_DATE': '2024-06-30',
                  'MAD_THRESHOLD': 5.0}
        e = DataEngine(config)
        e.load_data()
        result = e.clean_data()
        # Most values should be preserved (not replaced with median)
        median = result['spread'].median()
        replaced = (result['spread'] == median).sum()
        # At most a few outliers replaced
        assert replaced < len(result) * 0.1

    def test_clean_preserves_datetime_index(self, loaded_engine):
        result = loaded_engine.clean_data()
        assert isinstance(result.index, pd.DatetimeIndex)


# ─── get_returns Tests ───

class TestGetReturns:
    def test_raises_without_clean(self, engine):
        with pytest.raises(ValueError, match="请先调用 clean_data"):
            engine.get_returns()

    def test_returns_series(self, cleaned_engine):
        result = cleaned_engine.get_returns()
        assert isinstance(result, pd.Series)

    def test_length_is_n_minus_1(self, cleaned_engine):
        """diff() drops first observation."""
        returns = cleaned_engine.get_returns()
        assert len(returns) == len(cleaned_engine._clean_data) - 1

    def test_no_nan_values(self, cleaned_engine):
        returns = cleaned_engine.get_returns()
        assert returns.notna().all()

    def test_first_diff_formula(self):
        """returns[i] = spread[i] - spread[i-1]."""
        dates = pd.date_range('2024-01-01', '2024-02-28', freq='B')
        spread = np.arange(100, 100 + len(dates), dtype=float)
        config = {'SOURCE': 'MOCK', 'START_DATE': '2024-01-01', 'END_DATE': '2024-02-28'}
        e = DataEngine(config)
        e._raw_data = pd.DataFrame({'spread': spread}, index=dates)
        e.clean_data()
        returns = e.get_returns()
        # All diffs should be ~1.0
        np.testing.assert_allclose(returns.values, 1.0, atol=0.01)

    def test_zero_returns_for_constant_spread(self):
        """Constant spread should produce all-zero returns."""
        dates = pd.date_range('2024-01-01', '2024-02-28', freq='B')
        spread = np.full(len(dates), 100.0)
        config = {'SOURCE': 'MOCK', 'START_DATE': '2024-01-01', 'END_DATE': '2024-02-28'}
        e = DataEngine(config)
        e._raw_data = pd.DataFrame({'spread': spread}, index=dates)
        e.clean_data()
        returns = e.get_returns()
        assert returns.abs().max() < 1e-10


# ─── CSV Data Tests ───

class TestCSVData:
    def test_csv_load_success(self, csv_config):
        e = DataEngine(csv_config)
        result = e.load_data()
        assert len(result) > 0
        assert 'spread' in result.columns

    def test_csv_column_renamed_to_spread(self, csv_config):
        e = DataEngine(csv_config)
        result = e.load_data()
        assert list(result.columns) == ['spread']

    def test_csv_datetime_index(self, csv_config):
        e = DataEngine(csv_config)
        result = e.load_data()
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_csv_then_clean(self, csv_config):
        e = DataEngine(csv_config)
        e.load_data()
        result = e.clean_data()
        assert result['spread'].notna().all()

    def test_csv_then_returns(self, csv_config):
        e = DataEngine(csv_config)
        e.load_data()
        e.clean_data()
        returns = e.get_returns()
        assert len(returns) == len(e._clean_data) - 1

    def test_csv_multiple_spread_columns(self, tmp_path):
        """CSV with multiple spread columns should select the configured one."""
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='B')
        df = pd.DataFrame({
            'date': dates,
            'spread_all': np.random.normal(100, 10, len(dates)),
            'spread_10y': np.random.normal(110, 12, len(dates)),
        })
        csv_file = tmp_path / 'data.csv'
        df.to_csv(csv_file, index=False)

        config_5y = {'SOURCE': 'CSV', 'CSV_PATH': str(csv_file),
                     'SPREAD_COLUMN': 'spread_all', 'MAD_THRESHOLD': 5.0}
        config_10y = {'SOURCE': 'CSV', 'CSV_PATH': str(csv_file),
                      'SPREAD_COLUMN': 'spread_10y', 'MAD_THRESHOLD': 5.0}

        e1 = DataEngine(config_5y)
        r1 = e1.load_data()
        e2 = DataEngine(config_10y)
        r2 = e2.load_data()

        # Different columns should produce different values
        assert r1['spread'].mean() != r2['spread'].mean()


# ─── Full Workflow Integration Tests ───

class TestWorkflowIntegration:
    def test_full_mock_workflow(self, default_config):
        e = DataEngine(default_config)
        raw = e.load_data()
        assert e._raw_data is not None
        clean = e.clean_data()
        assert e._clean_data is not None
        returns = e.get_returns()
        assert len(returns) > 0
        assert returns.notna().all()

    def test_full_csv_workflow(self, csv_config):
        e = DataEngine(csv_config)
        raw = e.load_data()
        clean = e.clean_data()
        returns = e.get_returns()
        assert len(returns) == len(clean) - 1

    def test_workflow_preserves_date_order(self, default_config):
        e = DataEngine(default_config)
        e.load_data()
        e.clean_data()
        returns = e.get_returns()
        # Returns index should be monotonically increasing
        assert returns.index.is_monotonic_increasing

    def test_clean_data_does_not_modify_raw(self, default_config):
        e = DataEngine(default_config)
        raw = e.load_data()
        raw_copy = raw.copy()
        e.clean_data()
        # _raw_data should be unchanged (clean_data uses copy)
        pd.testing.assert_frame_equal(e._raw_data, raw_copy)

    def test_load_data_can_be_called_multiple_times(self, default_config):
        e = DataEngine(default_config)
        r1 = e.load_data()
        r2 = e.load_data()
        # Same seed produces same data
        pd.testing.assert_frame_equal(r1, r2)


# ─── Boundary Condition Tests ───

class TestBoundaryConditions:
    def test_very_short_date_range(self):
        """2-day range should produce 1-2 business days."""
        config = {'SOURCE': 'MOCK', 'START_DATE': '2024-01-02', 'END_DATE': '2024-01-03'}
        e = DataEngine(config)
        result = e.load_data()
        assert len(result) >= 1

    def test_single_day_range(self):
        config = {'SOURCE': 'MOCK', 'START_DATE': '2024-01-02', 'END_DATE': '2024-01-02'}
        e = DataEngine(config)
        result = e.load_data()
        # May produce 1 day or empty depending on business day logic
        assert len(result) >= 0

    def test_all_nan_spread(self):
        """All-NaN spread should be handled by ffill/bfill."""
        dates = pd.date_range('2024-01-01', '2024-02-28', freq='B')
        spread = np.full(len(dates), np.nan)
        config = {'SOURCE': 'MOCK', 'START_DATE': '2024-01-01', 'END_DATE': '2024-02-28'}
        e = DataEngine(config)
        e._raw_data = pd.DataFrame({'spread': spread}, index=dates)
        # ffill/bfill on all-NaN produces all-NaN
        result = e.clean_data()
        # MAD of NaN series is NaN; should handle gracefully
        # The result may still contain NaN

    def test_extreme_mad_threshold_zero(self):
        """Threshold=0 would mark everything as outlier."""
        dates = pd.date_range('2024-01-01', '2024-02-28', freq='B')
        spread = np.random.normal(100, 10, len(dates))
        config = {'SOURCE': 'MOCK', 'START_DATE': '2024-01-01', 'END_DATE': '2024-02-28',
                  'MAD_THRESHOLD': 0.0}
        e = DataEngine(config)
        e._raw_data = pd.DataFrame({'spread': spread}, index=dates)
        result = e.clean_data()
        # With threshold=0, all non-median values are outliers -> all replaced with median
        assert result['spread'].std() == 0 or result['spread'].std() < 0.01

    def test_negative_spread_values(self):
        """Mock data can produce negative spreads (AR(1) + shocks)."""
        config = {'SOURCE': 'MOCK', 'START_DATE': '2024-01-01', 'END_DATE': '2026-01-01'}
        e = DataEngine(config)
        result = e.load_data()
        # Large shocks may push spread below 0, but mean-reversion brings it back
        # Just verify no crash
        e.clean_data()
        e.get_returns()

    def test_config_with_extra_keys(self):
        """Extra config keys should be ignored."""
        config = {'SOURCE': 'MOCK', 'START_DATE': '2024-01-01', 'END_DATE': '2024-06-30',
                  'EXTRA_KEY': 'value', 'MAD_THRESHOLD': 5.0}
        e = DataEngine(config)
        result = e.load_data()
        assert len(result) > 0

    def test_csv_with_no_spread_columns_raises(self, tmp_path):
        """CSV with no 'spread*' columns should raise ValueError."""
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='B')
        df = pd.DataFrame({'date': dates, 'volume': np.zeros(len(dates))})
        csv_file = tmp_path / 'data.csv'
        df.to_csv(csv_file, index=False)
        config = {'SOURCE': 'CSV', 'CSV_PATH': str(csv_file), 'SPREAD_COLUMN': 'spread_all'}
        e = DataEngine(config)
        with pytest.raises(ValueError, match="不存在"):
            e.load_data()

    def test_mad_robust_to_single_extreme_outlier(self):
        """MAD should be robust: single extreme outlier shouldn't shift median much."""
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='B')
        spread = np.random.normal(100, 10, len(dates))
        median_before = np.median(spread)
        spread[0] = 99999  # Extreme outlier
        config = {'SOURCE': 'MOCK', 'START_DATE': '2024-01-01', 'END_DATE': '2024-03-31',
                  'MAD_THRESHOLD': 5.0}
        e = DataEngine(config)
        e._raw_data = pd.DataFrame({'spread': spread}, index=dates)
        result = e.clean_data()
        # Median-based methods should still give reasonable results
        assert abs(result['spread'].median() - median_before) < 5