"""
Random Forest Regression Model
"""
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor


@dataclass
class RandomForestConfig:
    """Random Forest configuration"""
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str = 'sqrt'


class RandomForestModel:
    """Random Forest model for regression"""
    
    def __init__(self, config: RandomForestConfig = None):
        self.config = config or RandomForestConfig()
        self.model = None
        self.is_fitted = False
        self.feature_names = None
    
    def _create_model(self) -> None:
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            random_state=42,
            n_jobs=-1
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train Random Forest model"""
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
        
        if self.feature_names:
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        return pd.DataFrame({'importance': importance})
