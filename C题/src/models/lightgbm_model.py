"""
LightGBM Regression Model
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


@dataclass
class LightGBMConfig:
    """LightGBM configuration"""
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 500
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0


class LightGBMModel:
    """LightGBM model for regression"""
    
    def __init__(self, config: LightGBMConfig = None):
        self.config = config or LightGBMConfig()
        self.model = None
        self.is_fitted = False
        self.feature_names = None
    
    def _create_model(self) -> None:
        if not LGBM_AVAILABLE:
            raise ImportError("lightgbm is required")
        
        self.model = LGBMRegressor(
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            num_leaves=self.config.num_leaves,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=42,
            verbose=-1
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train LightGBM model"""
        self._create_model()
        self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance = self.model.feature_importances_
        total = importance.sum()
        normalized = importance / total if total > 0 else importance
        
        if self.feature_names:
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': normalized
            }).sort_values('importance', ascending=False)
        
        return pd.DataFrame({'importance': normalized})
