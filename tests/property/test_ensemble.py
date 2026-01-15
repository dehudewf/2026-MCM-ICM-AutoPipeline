"""
Property-based tests for ensemble models
Feature: olympic-medal-prediction, Properties 13-14
"""
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.ensemble import EnsemblePredictor, EnsembleConfig


class TestWeightedAverageCorrectness:
    """
    Feature: olympic-medal-prediction, Property 13: Weighted Average Correctness
    
    For any set of model predictions with weights that sum to 1,
    the ensemble prediction should equal the weighted sum.
    """
    
    @given(
        n=st.integers(min_value=5, max_value=50),
        w1=st.floats(min_value=0.1, max_value=0.5),
        w2=st.floats(min_value=0.1, max_value=0.5)
    )
    @settings(max_examples=100)
    def test_weighted_average_calculation(self, n, w1, w2):
        """Weighted average should equal manual calculation"""
        w3 = 1.0 - w1 - w2
        if w3 < 0:
            w3 = 0.1
            total = w1 + w2 + w3
            w1, w2, w3 = w1/total, w2/total, w3/total
        
        np.random.seed(42)
        pred1 = np.random.uniform(10, 100, n)
        pred2 = np.random.uniform(10, 100, n)
        pred3 = np.random.uniform(10, 100, n)
        
        predictions = {'m1': pred1, 'm2': pred2, 'm3': pred3}
        weights = {'m1': w1, 'm2': w2, 'm3': w3}
        
        ensemble = EnsemblePredictor()
        result = ensemble.weighted_average(predictions, weights)
        
        # Manual calculation
        expected = (w1 * pred1 + w2 * pred2 + w3 * pred3) / (w1 + w2 + w3)
        
        assert np.allclose(result, expected, rtol=1e-5)


class TestHighVarianceFlagging:
    """
    Feature: olympic-medal-prediction, Property 14: High Variance Flagging
    
    For any set of model predictions with standard deviation exceeding
    a threshold, the system should flag these predictions.
    """
    
    @given(n=st.integers(min_value=10, max_value=50))
    @settings(max_examples=50)
    def test_high_variance_detected(self, n):
        """High variance predictions should be flagged"""
        np.random.seed(42)
        
        # Create predictions with known high variance at specific indices
        base = np.random.uniform(50, 100, n)
        
        # Model 1: close to base
        pred1 = base + np.random.normal(0, 1, n)
        
        # Model 2: close to base
        pred2 = base + np.random.normal(0, 1, n)
        
        # Model 3: very different at index 0
        pred3 = base.copy()
        pred3[0] = base[0] * 2  # 100% different
        
        predictions = {'m1': pred1, 'm2': pred2, 'm3': pred3}
        
        config = EnsembleConfig(variance_threshold=0.20)
        ensemble = EnsemblePredictor(config)
        
        high_var_indices = ensemble.detect_high_variance(predictions)
        
        # Index 0 should be flagged due to high variance
        assert 0 in high_var_indices
    
    @given(n=st.integers(min_value=10, max_value=50))
    @settings(max_examples=50)
    def test_low_variance_not_flagged(self, n):
        """Low variance predictions should not be flagged"""
        np.random.seed(42)
        
        # Create predictions with low variance
        base = np.random.uniform(50, 100, n)
        
        # All models very close
        pred1 = base + np.random.normal(0, 0.1, n)
        pred2 = base + np.random.normal(0, 0.1, n)
        pred3 = base + np.random.normal(0, 0.1, n)
        
        predictions = {'m1': pred1, 'm2': pred2, 'm3': pred3}
        
        config = EnsembleConfig(variance_threshold=0.20)
        ensemble = EnsemblePredictor(config)
        
        high_var_indices = ensemble.detect_high_variance(predictions)
        
        # Should have few or no flagged indices
        assert len(high_var_indices) < n * 0.1  # Less than 10%
