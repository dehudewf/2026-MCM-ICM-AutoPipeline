"""
Property-based tests for optimization modules
Feature: olympic-medal-prediction, Properties 15-20
"""
import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from optimization.host_effect import HostEffectAdjuster
from optimization.event_change import EventChangeAdjuster
from optimization.bayesian import BayesianUncertainty


class TestHostEffectMultiplicativeAdjustment:
    """
    Feature: olympic-medal-prediction, Property 15: Host Effect Multiplicative Adjustment
    
    For any baseline prediction P and host effect rate r, the adjusted prediction
    should equal P × (1 + r).
    """
    
    @given(
        baseline=st.floats(min_value=10, max_value=200, allow_nan=False),
        effect_rate=st.floats(min_value=0.1, max_value=0.3, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_multiplicative_adjustment(self, baseline, effect_rate):
        """Adjusted prediction should equal baseline * (1 + rate)"""
        adjuster = HostEffectAdjuster()
        result = adjuster.apply_multiplicative_adjustment(baseline, effect_rate)
        expected = baseline * (1 + effect_rate)
        assert abs(result - expected) < 1e-6


class TestEventAdditionImpact:
    """
    Feature: olympic-medal-prediction, Property 16: Event Addition Impact
    
    For any country and newly added event, the estimated additional medals
    should be non-negative.
    """
    
    @given(n_new_events=st.integers(min_value=0, max_value=10))
    @settings(max_examples=50)
    def test_event_addition_non_negative(self, n_new_events):
        """Event addition impact should be non-negative"""
        adjuster = EventChangeAdjuster()
        adjuster.country_event_rates = {'USA': 0.15}
        
        new_events = [f'event_{i}' for i in range(n_new_events)]
        impact = adjuster.estimate_event_impact('USA', new_events=new_events)
        
        assert impact >= 0


class TestEventRemovalImpact:
    """
    Feature: olympic-medal-prediction, Property 17: Event Removal Impact
    
    For any removed event, the adjusted medal prediction should be less than
    or equal to the original prediction.
    """
    
    @given(
        baseline=st.floats(min_value=50, max_value=150, allow_nan=False),
        n_removed=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=50)
    def test_event_removal_decreases_prediction(self, baseline, n_removed):
        """Removing events should decrease or maintain prediction"""
        adjuster = EventChangeAdjuster()
        adjuster.country_event_rates = {'USA': 0.1}
        
        removed_events = [f'event_{i}' for i in range(n_removed)]
        impact = adjuster.estimate_event_impact('USA', removed_events=removed_events)
        
        # Impact should be negative (decrease)
        assert impact <= 0
        
        # Adjusted should be <= original
        adjusted = max(0, baseline + impact)
        assert adjusted <= baseline


class TestCountrySpecificEventRates:
    """
    Feature: olympic-medal-prediction, Property 18: Country-Specific Event Rates
    
    For any country, the medal-per-event rate should be calculated from
    that country's historical data.
    """
    
    @given(
        usa_rate=st.floats(min_value=0.05, max_value=0.3, allow_nan=False),
        chn_rate=st.floats(min_value=0.05, max_value=0.3, allow_nan=False)
    )
    @settings(max_examples=50)
    def test_country_specific_rates(self, usa_rate, chn_rate):
        """Different countries should have different rates"""
        adjuster = EventChangeAdjuster()
        adjuster.country_event_rates = {'USA': usa_rate, 'CHN': chn_rate}
        
        usa_impact = adjuster.estimate_event_impact('USA', new_events=['event1'])
        chn_impact = adjuster.estimate_event_impact('CHN', new_events=['event1'])
        
        # Impacts should reflect country-specific rates
        assert abs(usa_impact - usa_rate) < 1e-6
        assert abs(chn_impact - chn_rate) < 1e-6


class TestConfidenceIntervalCoverage:
    """
    Feature: olympic-medal-prediction, Property 19: Confidence Interval Coverage
    
    For any prediction with 95% CI [L, U], the point estimate should satisfy
    L <= estimate <= U.
    """
    
    @given(n=st.integers(min_value=10, max_value=50))
    @settings(max_examples=50)
    def test_point_estimate_within_interval(self, n):
        """Point estimate should be within confidence interval"""
        np.random.seed(42)
        
        # Create predictions from multiple models
        base = np.random.uniform(50, 100, n)
        predictions = {
            'm1': base + np.random.normal(0, 5, n),
            'm2': base + np.random.normal(0, 5, n),
            'm3': base + np.random.normal(0, 5, n)
        }
        
        uncertainty = BayesianUncertainty()
        result = uncertainty.compute_confidence_intervals(predictions)
        
        # Point estimate should be within bounds
        assert np.all(result.point_estimate >= result.lower_bound - 1e-6)
        assert np.all(result.point_estimate <= result.upper_bound + 1e-6)


class TestWideIntervalFlagging:
    """
    Feature: olympic-medal-prediction, Property 20: Wide Interval Flagging
    
    For any prediction where (CI_upper - CI_lower) > 0.2 × point_estimate,
    the system should flag this prediction as having high uncertainty.
    """
    
    @given(n=st.integers(min_value=10, max_value=30))
    @settings(max_examples=50)
    def test_wide_intervals_flagged(self, n):
        """Wide confidence intervals should be flagged"""
        np.random.seed(42)
        
        # Create predictions with known high variance at index 0
        base = np.random.uniform(50, 100, n)
        
        pred1 = base.copy()
        pred2 = base.copy()
        pred3 = base.copy()
        
        # Make index 0 have very high variance
        pred1[0] = base[0] * 0.5
        pred2[0] = base[0] * 1.0
        pred3[0] = base[0] * 1.5
        
        predictions = {'m1': pred1, 'm2': pred2, 'm3': pred3}
        
        uncertainty = BayesianUncertainty(wide_ci_threshold=0.20)
        result = uncertainty.compute_confidence_intervals(predictions)
        
        # Index 0 should be flagged
        assert 0 in result.high_uncertainty_indices
    
    @given(n=st.integers(min_value=10, max_value=30))
    @settings(max_examples=50)
    def test_narrow_intervals_not_flagged(self, n):
        """Narrow confidence intervals should not be flagged"""
        np.random.seed(42)
        
        # Create predictions with low variance
        base = np.random.uniform(50, 100, n)
        
        predictions = {
            'm1': base + np.random.normal(0, 0.1, n),
            'm2': base + np.random.normal(0, 0.1, n),
            'm3': base + np.random.normal(0, 0.1, n)
        }
        
        uncertainty = BayesianUncertainty(wide_ci_threshold=0.20)
        result = uncertainty.compute_confidence_intervals(predictions)
        
        # Should have few or no flagged indices
        assert len(result.high_uncertainty_indices) < n * 0.2
