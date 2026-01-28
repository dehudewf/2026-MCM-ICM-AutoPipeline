"""
Property-based tests for ML models
Feature: olympic-medal-prediction, Properties 11-12
"""
import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestFeatureImportanceNormalization:
    """
    Feature: olympic-medal-prediction, Property 11: Feature Importance Normalization
    
    For any trained tree-based model, the extracted feature importance scores
    should be non-negative and sum to approximately 1.0.
    """
    
    @given(
        n_samples=st.integers(min_value=50, max_value=200),
        n_features=st.integers(min_value=3, max_value=10)
    )
    @settings(max_examples=20)
    def test_xgboost_importance_sums_to_one(self, n_samples, n_features):
        """XGBoost feature importance should sum to ~1"""
        try:
            import xgboost
        except ImportError:
            pytest.skip("xgboost not installed")
        
        from models.xgboost_model import XGBoostModel, XGBoostConfig
        
        # Generate random data
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randn(n_samples))
        
        # Train model with minimal config
        config = XGBoostConfig(n_estimators=10, max_depth=3)
        model = XGBoostModel(config)
        
        try:
            model.fit(X, y)
        except ImportError:
            pytest.skip("xgboost not available")
        
        # Get importance
        importance = model.get_feature_importance()
        
        # Check properties
        assert all(importance['importance'] >= 0)
        assert abs(importance['importance'].sum() - 1.0) < 0.01


class TestEvaluationMetricsValidity:
    """
    Feature: olympic-medal-prediction, Property 12: Evaluation Metrics Validity
    
    For any set of predictions and actual values, computed metrics should satisfy:
    MAE >= 0, RMSE >= MAE, MAPE >= 0, and R² <= 1.
    """
    
    @given(
        n=st.integers(min_value=10, max_value=100),
        noise=st.floats(min_value=0.1, max_value=10.0)
    )
    @settings(max_examples=100)
    def test_metric_bounds(self, n, noise):
        """Metrics should satisfy mathematical bounds"""
        from validation.evaluator import ModelEvaluator
        
        np.random.seed(42)
        y_true = np.random.uniform(10, 100, n)
        y_pred = y_true + np.random.normal(0, noise, n)
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred)
        
        # MAE >= 0
        assert metrics.mae >= 0
        
        # RMSE >= MAE (always true mathematically)
        assert metrics.rmse >= metrics.mae - 1e-6
        
        # MAPE >= 0
        assert metrics.mape >= 0
        
        # R² <= 1
        assert metrics.r2 <= 1.0 + 1e-6
    
    @given(n=st.integers(min_value=10, max_value=50))
    @settings(max_examples=50)
    def test_perfect_prediction_metrics(self, n):
        """Perfect predictions should have MAE=0, RMSE=0, R²=1"""
        from validation.evaluator import ModelEvaluator
        
        np.random.seed(42)
        y_true = np.random.uniform(10, 100, n)
        y_pred = y_true.copy()  # Perfect prediction
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_true, y_pred)
        
        assert abs(metrics.mae) < 1e-6
        assert abs(metrics.rmse) < 1e-6
        assert abs(metrics.r2 - 1.0) < 1e-6
