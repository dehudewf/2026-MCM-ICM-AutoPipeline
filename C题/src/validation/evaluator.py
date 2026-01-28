"""
Model Evaluation Module
Implements evaluation metrics for model performance
"""
import numpy as np
import pandas as pd
from typing import Dict
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    mae: float
    rmse: float
    mape: float
    r2: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'mae': self.mae,
            'rmse': self.rmse,
            'mape': self.mape,
            'r2': self.r2
        }


class ModelEvaluator:
    """
    Evaluates model predictions using various metrics.
    
    Metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Square Error
    - MAPE: Mean Absolute Percentage Error
    - R²: Coefficient of Determination
    """
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² (coefficient of determination)"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)

    def evaluate(self, y_true: np.ndarray, 
                 y_pred: np.ndarray) -> EvaluationMetrics:
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            EvaluationMetrics with all metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        return EvaluationMetrics(
            mae=self.mae(y_true, y_pred),
            rmse=self.rmse(y_true, y_pred),
            mape=self.mape(y_true, y_pred),
            r2=self.r2(y_true, y_pred)
        )
    
    def validate_metrics(self, metrics: EvaluationMetrics) -> Dict[str, bool]:
        """
        Validate that metrics are within expected bounds.
        
        Returns:
            Dictionary of validation results
        """
        return {
            'mae_valid': metrics.mae >= 0,
            'rmse_valid': metrics.rmse >= metrics.mae,  # RMSE >= MAE always
            'mape_valid': metrics.mape >= 0,
            'r2_valid': metrics.r2 <= 1.0
        }
    
    def compare_models(self, results: Dict[str, EvaluationMetrics]) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            results: Dictionary mapping model names to metrics
            
        Returns:
            DataFrame with comparison
        """
        data = []
        for name, metrics in results.items():
            data.append({
                'model': name,
                **metrics.to_dict()
            })
        return pd.DataFrame(data).sort_values('mae')
