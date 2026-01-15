"""
XGBoost Regression Model
Implements XGBoost for Olympic medal prediction
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


@dataclass
class XGBoostConfig:
    """XGBoost model configuration"""
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 500
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50


class XGBoostModel:
    """
    XGBoost model for regression.
    
    Features:
    - Gradient boosting
    - Feature importance extraction
    - Early stopping
    """
    
    def __init__(self, config: XGBoostConfig = None):
        """Initialize XGBoost model"""
        self.config = config or XGBoostConfig()
        self.model = None
        self.is_fitted = False
        self.feature_names = None

    def _create_model(self) -> None:
        """Create XGBoost model with config parameters"""
        if not XGB_AVAILABLE:
            raise ImportError("xgboost is required")
        
        self.model = XGBRegressor(
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            gamma=self.config.gamma,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=42
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            eval_set: List[Tuple] = None) -> None:
        """
        Train XGBoost model.
        
        Args:
            X: Feature matrix
            y: Target values
            eval_set: List of (X, y) tuples for evaluation
        """
        self._create_model()
        self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        
        fit_params = {}
        if eval_set:
            fit_params['eval_set'] = eval_set
            fit_params['verbose'] = False
        
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance = self.model.feature_importances_
        
        if self.feature_names:
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        return pd.DataFrame({'importance': importance})
