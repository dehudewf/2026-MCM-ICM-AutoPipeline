"""
Property-based tests for validation module
Feature: olympic-medal-prediction, Properties 21-22
"""
import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from validation.backtest import BacktestValidator


@st.composite
def temporal_dataframe(draw):
    """Generate DataFrame with year column"""
    years = list(range(1996, 2024, 4))  # Olympic years
    n_countries = draw(st.integers(min_value=3, max_value=10))
    
    rows = []
    for year in years:
        for i in range(n_countries):
            rows.append({
                'year': year,
                'country': f'C{i}',
                'total': draw(st.integers(min_value=0, max_value=100))
            })
    
    return pd.DataFrame(rows)


class TestTemporalDataLeakagePrevention:
    """
    Feature: olympic-medal-prediction, Property 21: Temporal Data Leakage Prevention
    
    For any backtesting scenario predicting year Y, the training data should
    contain only records where year < Y.
    """
    
    @given(df=temporal_dataframe())
    @settings(max_examples=50)
    def test_no_future_data_in_training(self, df):
        """Training data should not contain future years"""
        validator = BacktestValidator()
        
        test_year = 2020
        train_df, test_df = validator.create_temporal_split(df, test_year)
        
        # All training years should be < test year
        if len(train_df) > 0:
            assert train_df['year'].max() < test_year
        
        # All test years should be == test year
        if len(test_df) > 0:
            assert test_df['year'].min() == test_year
            assert test_df['year'].max() == test_year
    
    @given(df=temporal_dataframe())
    @settings(max_examples=50)
    def test_validate_no_leakage(self, df):
        """Leakage validation should pass for correct splits"""
        validator = BacktestValidator()
        
        test_year = 2016
        train_df, test_df = validator.create_temporal_split(df, test_year)
        
        # Should pass validation
        assert validator.validate_no_leakage(train_df, test_df)


class TestCrossValidationMetricAggregation:
    """
    Feature: olympic-medal-prediction, Property 22: Cross-Validation Metric Aggregation
    
    For any k-fold cross-validation, the reported average metric should equal
    the arithmetic mean of the metric values across all k folds.
    """
    
    @given(
        n_folds=st.integers(min_value=2, max_value=10),
        mae_values=st.lists(
            st.floats(min_value=1, max_value=50, allow_nan=False),
            min_size=2, max_size=10
        )
    )
    @settings(max_examples=100)
    def test_metric_aggregation_equals_mean(self, n_folds, mae_values):
        """Aggregated metric should equal arithmetic mean"""
        # Create fold metrics
        fold_metrics = [{'mae': v} for v in mae_values[:n_folds]]
        
        validator = BacktestValidator()
        aggregated = validator.aggregate_cv_metrics(fold_metrics)
        
        # Calculate expected mean
        expected_mean = np.mean([m['mae'] for m in fold_metrics])
        
        # Should match
        assert abs(aggregated['mae_mean'] - expected_mean) < 1e-6
    
    @given(
        values=st.lists(
            st.floats(min_value=1, max_value=100, allow_nan=False),
            min_size=3, max_size=10
        )
    )
    @settings(max_examples=50)
    def test_std_calculation(self, values):
        """Standard deviation should be correctly calculated"""
        fold_metrics = [{'rmse': v} for v in values]
        
        validator = BacktestValidator()
        aggregated = validator.aggregate_cv_metrics(fold_metrics)
        
        expected_std = np.std(values)
        
        assert abs(aggregated['rmse_std'] - expected_std) < 1e-6


from validation.sensitivity import SensitivityAnalyzer


class TestSensitivityScenarioGeneration:
    """
    Feature: olympic-medal-prediction, Property 23: Sensitivity Scenario Generation
    
    For any baseline feature value V and variation percentage p, the sensitivity
    analysis should test scenarios with values V × (1 - p) and V × (1 + p).
    """
    
    @given(
        base_value=st.floats(min_value=10, max_value=1000, allow_nan=False),
        variation_pct=st.floats(min_value=0.01, max_value=0.20, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_scenario_generation(self, base_value, variation_pct):
        """Scenarios should be generated at correct variation levels"""
        analyzer = SensitivityAnalyzer()
        lower, upper = analyzer.generate_scenario_variations(base_value, variation_pct)
        
        expected_lower = base_value * (1 - variation_pct)
        expected_upper = base_value * (1 + variation_pct)
        
        assert abs(lower - expected_lower) < 1e-6
        assert abs(upper - expected_upper) < 1e-6


class TestHighSensitivityFlagging:
    """
    Feature: olympic-medal-prediction, Property 24: High Sensitivity Flagging
    
    For any feature where a 10% input change causes >10% prediction change,
    the system should flag this feature as having high sensitivity.
    """
    
    def test_high_sensitivity_flagged(self):
        """Features with high sensitivity should be flagged"""
        from validation.sensitivity import SensitivityResult
        
        analyzer = SensitivityAnalyzer()
        analyzer.high_sensitivity_threshold = 0.10
        
        # Create results with known sensitivities
        results = {
            'high_sens': SensitivityResult(
                feature='high_sens',
                variations=[0.9, 1.0, 1.1],
                predictions=[90, 100, 115],  # 25% change for 20% input change
                base_prediction=100,
                sensitivity_score=0.25  # > 0.10 threshold
            ),
            'low_sens': SensitivityResult(
                feature='low_sens',
                variations=[0.9, 1.0, 1.1],
                predictions=[99, 100, 101],  # 2% change for 20% input change
                base_prediction=100,
                sensitivity_score=0.02  # < 0.10 threshold
            )
        }
        
        flagged = analyzer.flag_high_sensitivity(results)
        
        assert 'high_sens' in flagged
        assert 'low_sens' not in flagged
