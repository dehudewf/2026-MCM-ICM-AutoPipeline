"""
Ensemble Model Module
Implements model fusion and stacking
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from sklearn.linear_model import Ridge


@dataclass
class EnsembleConfig:
    """Ensemble configuration with model weights"""
    # Time series model weights
    arima_weight: float = 0.15
    prophet_weight: float = 0.15
    lstm_weight: float = 0.20
    
    # ML model weights
    xgboost_weight: float = 0.25
    lightgbm_weight: float = 0.15
    random_forest_weight: float = 0.10
    
    # Variance threshold for flagging
    variance_threshold: float = 0.20
    
    def get_weights(self) -> Dict[str, float]:
        return {
            'arima': self.arima_weight,
            'prophet': self.prophet_weight,
            'lstm': self.lstm_weight,
            'xgboost': self.xgboost_weight,
            'lightgbm': self.lightgbm_weight,
            'random_forest': self.random_forest_weight
        }
    
    def validate_weights(self) -> bool:
        """Check if weights sum to 1.0"""
        total = sum(self.get_weights().values())
        return abs(total - 1.0) < 0.01


class EnsemblePredictor:
    """
    Combines predictions from multiple models using weighted averaging.
    """
    
    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.models = {}
        self.weights = self.config.get_weights()

    def add_model(self, name: str, model: Any, weight: float = None) -> None:
        """Add a model to the ensemble"""
        self.models[name] = model
        if weight is not None:
            self.weights[name] = weight
    
    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute weighted average of predictions.
        
        Args:
            predictions: Dictionary mapping model names to prediction arrays
            
        Returns:
            Weighted average predictions
        """
        # Normalize weights for available models
        available_weights = {k: v for k, v in self.weights.items() if k in predictions}
        total_weight = sum(available_weights.values())
        
        if total_weight == 0:
            raise ValueError("No valid predictions provided")
        
        # Compute weighted average
        result = None
        for name, pred in predictions.items():
            if name in available_weights:
                weight = available_weights[name] / total_weight
                if result is None:
                    result = weight * np.array(pred)
                else:
                    result += weight * np.array(pred)
        
        return result
    
    def weighted_average(self, predictions: Dict[str, np.ndarray],
                         weights: Dict[str, float]) -> np.ndarray:
        """
        Compute weighted average with custom weights.
        """
        total_weight = sum(weights.values())
        if total_weight == 0:
            raise ValueError("Weights sum to zero")
        
        result = None
        for name, pred in predictions.items():
            if name in weights:
                w = weights[name] / total_weight
                if result is None:
                    result = w * np.array(pred)
                else:
                    result += w * np.array(pred)
        
        return result
    
    def detect_high_variance(self, predictions: Dict[str, np.ndarray]) -> List[int]:
        """
        Detect predictions with high variance across models.
        
        Returns:
            List of indices with high variance
        """
        pred_array = np.array(list(predictions.values()))
        mean_pred = np.mean(pred_array, axis=0)
        std_pred = np.std(pred_array, axis=0)
        
        # Flag where std > threshold * mean
        threshold = self.config.variance_threshold
        high_var_mask = std_pred > threshold * np.abs(mean_pred)
        
        return np.where(high_var_mask)[0].tolist()


class StackingEnsemble:
    """
    Stacking ensemble with meta-learner.
    
    Uses base model predictions as features for a meta-model.
    """
    
    def __init__(self, base_models: List[Tuple[str, Any]] = None,
                 meta_model: Any = None):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: List of (name, model) tuples
            meta_model: Meta-learner model (default: Ridge)
        """
        self.base_models = base_models or []
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> None:
        """
        Fit stacking ensemble.
        
        Args:
            X: Feature matrix
            y: Target values
            cv: Number of cross-validation folds
        """
        from sklearn.model_selection import KFold
        
        n_samples = len(X)
        meta_features = np.zeros((n_samples, len(self.base_models)))
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Generate out-of-fold predictions for each base model
        for i, (name, model) in enumerate(self.base_models):
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y.iloc[train_idx]
                
                model.fit(X_train, y_train)
                meta_features[val_idx, i] = model.predict(X_val)
        
        # Fit meta-model
        self.meta_model.fit(meta_features, y)
        
        # Refit base models on full data
        for name, model in self.base_models:
            model.fit(X, y)
        
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using stacking"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get base model predictions
        meta_features = np.zeros((len(X), len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models):
            meta_features[:, i] = model.predict(X)
        
        # Meta-model prediction
        return self.meta_model.predict(meta_features)
