"""
Property-based tests for analysis modules
Feature: olympic-medal-prediction, Properties 25-27
"""
import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from analysis.regional import RegionalAnalyzer


class TestRegionalAggregationConsistency:
    """
    Feature: olympic-medal-prediction, Property 25: Regional Aggregation Consistency
    
    For any region, the sum of country-level predictions should equal
    the region-level aggregated forecast.
    """
    
    @given(
        n_countries=st.integers(min_value=2, max_value=10),
        predictions=st.lists(
            st.floats(min_value=0, max_value=100, allow_nan=False),
            min_size=2, max_size=10
        )
    )
    @settings(max_examples=100)
    def test_aggregation_equals_sum(self, n_countries, predictions):
        """Regional total should equal sum of country predictions"""
        # Create predictions dict
        countries = [f'C{i}' for i in range(min(n_countries, len(predictions)))]
        pred_dict = {c: p for c, p in zip(countries, predictions)}
        
        analyzer = RegionalAnalyzer()
        regional_total = analyzer.aggregate_regional_predictions(pred_dict, countries)
        
        expected_sum = sum(predictions[:len(countries)])
        
        assert abs(regional_total - expected_sum) < 1e-6
    
    @given(
        predictions=st.dictionaries(
            st.sampled_from(['BRA', 'ARG', 'COL', 'CHI', 'PER']),
            st.floats(min_value=0, max_value=50, allow_nan=False),
            min_size=1, max_size=5
        )
    )
    @settings(max_examples=50)
    def test_regional_prediction_consistency(self, predictions):
        """Regional prediction total should match sum of country predictions"""
        analyzer = RegionalAnalyzer()
        result = analyzer.create_regional_prediction(predictions, 'South America')
        
        expected_total = sum(result.country_predictions.values())
        
        assert abs(result.regional_total - expected_total) < 1e-6


from analysis.event_impact import EventImpactAnalyzer


class TestCorrelationBounds:
    """
    Feature: olympic-medal-prediction, Property 26: Correlation Bounds
    
    For any two variables, the computed Pearson and Spearman correlation
    coefficients should be in the range [-1, 1].
    """
    
    @given(
        n=st.integers(min_value=10, max_value=100),
        noise=st.floats(min_value=0.1, max_value=10, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_pearson_within_bounds(self, n, noise):
        """Pearson correlation should be in [-1, 1]"""
        np.random.seed(42)
        
        events = np.random.uniform(10, 50, n)
        medals = events * 2 + np.random.normal(0, noise, n)
        
        analyzer = EventImpactAnalyzer()
        result = analyzer.compute_correlations(events, medals)
        
        assert -1 <= result.pearson_r <= 1
        assert analyzer.validate_correlation_bounds(result.pearson_r)
    
    @given(
        n=st.integers(min_value=10, max_value=100),
        noise=st.floats(min_value=0.1, max_value=10, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_spearman_within_bounds(self, n, noise):
        """Spearman correlation should be in [-1, 1]"""
        np.random.seed(42)
        
        events = np.random.uniform(10, 50, n)
        medals = events * 2 + np.random.normal(0, noise, n)
        
        analyzer = EventImpactAnalyzer()
        result = analyzer.compute_correlations(events, medals)
        
        assert -1 <= result.spearman_r <= 1
        assert analyzer.validate_correlation_bounds(result.spearman_r)


from analysis.coach_effect import CoachEffectAnalyzer


class TestStatisticalTestValidity:
    """
    Feature: olympic-medal-prediction, Property 27: Statistical Test Validity
    
    For any two samples (before/after coach arrival), the t-test should return
    a valid p-value in the range [0, 1] and Cohen's d should be calculable.
    """
    
    @given(
        n_before=st.integers(min_value=3, max_value=20),
        n_after=st.integers(min_value=3, max_value=20),
        mean_diff=st.floats(min_value=-10, max_value=10, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_p_value_in_valid_range(self, n_before, n_after, mean_diff):
        """P-value should be in [0, 1]"""
        np.random.seed(42)
        
        before = np.random.uniform(10, 30, n_before)
        after = np.random.uniform(10 + mean_diff, 30 + mean_diff, n_after)
        
        analyzer = CoachEffectAnalyzer()
        result = analyzer.perform_t_test(before, after)
        
        assert 0 <= result.p_value <= 1
    
    @given(
        n_before=st.integers(min_value=3, max_value=20),
        n_after=st.integers(min_value=3, max_value=20)
    )
    @settings(max_examples=100)
    def test_cohens_d_calculable(self, n_before, n_after):
        """Cohen's d should be calculable for any valid samples"""
        np.random.seed(42)
        
        before = np.random.uniform(10, 30, n_before)
        after = np.random.uniform(15, 35, n_after)
        
        analyzer = CoachEffectAnalyzer()
        result = analyzer.compute_cohens_d(before, after)
        
        # Should return a valid result
        assert result.cohens_d is not None
        assert result.interpretation in ['small', 'medium', 'large', 'none']
    
    @given(
        n=st.integers(min_value=5, max_value=20)
    )
    @settings(max_examples=50)
    def test_effect_size_interpretation(self, n):
        """Effect size should have correct interpretation"""
        np.random.seed(42)
        
        # Create samples with known large effect
        before = np.random.uniform(10, 15, n)
        after = np.random.uniform(25, 30, n)  # Much higher
        
        analyzer = CoachEffectAnalyzer()
        result = analyzer.compute_cohens_d(before, after)
        
        # Large difference should give large effect size
        assert result.interpretation in ['medium', 'large']
