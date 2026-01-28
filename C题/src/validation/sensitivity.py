"""
Sensitivity Analysis Module
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis"""
    feature: str
    variations: List[float]
    predictions: List[float]
    base_prediction: float
    sensitivity_score: float  # % change in prediction per % change in input


class SensitivityAnalyzer:
    """
    Analyzes model sensitivity to input variations.
    
    Tests how predictions change with:
    - GDP variations (±2%)
    - Investment variations (±10%)
    - Event count variations (±5)
    - Host effect variations (±5%)
    """
    
    def __init__(self):
        self.high_sensitivity_threshold = 0.10  # 10%
    
    def vary_feature(self, X: pd.DataFrame,
                     feature: str,
                     variation_range: np.ndarray) -> List[pd.DataFrame]:
        """
        Create variations of a feature.
        
        Args:
            X: Original feature matrix
            feature: Feature to vary
            variation_range: Array of multipliers (e.g., [0.98, 1.0, 1.02])
            
        Returns:
            List of DataFrames with varied feature values
        """
        variations = []
        
        for mult in variation_range:
            X_varied = X.copy()
            X_varied[feature] = X_varied[feature] * mult
            variations.append(X_varied)
        
        return variations
    
    def compute_sensitivity(self, model: Any,
                            X: pd.DataFrame,
                            feature: str,
                            variation_pct: float = 0.10) -> SensitivityResult:
        """
        Compute sensitivity of predictions to a feature.
        
        Args:
            model: Trained model with predict method
            X: Feature matrix
            feature: Feature to analyze
            variation_pct: Variation percentage (e.g., 0.10 for ±10%)
        """
        # Base prediction
        base_pred = model.predict(X).mean()
        
        # Varied predictions
        variations = [1 - variation_pct, 1.0, 1 + variation_pct]
        predictions = []
        
        for mult in variations:
            X_varied = X.copy()
            X_varied[feature] = X_varied[feature] * mult
            pred = model.predict(X_varied).mean()
            predictions.append(pred)
        
        # Calculate sensitivity score
        pred_change = (predictions[2] - predictions[0]) / (base_pred + 1e-6)
        input_change = 2 * variation_pct
        sensitivity = abs(pred_change / input_change) if input_change > 0 else 0
        
        return SensitivityResult(
            feature=feature,
            variations=variations,
            predictions=predictions,
            base_prediction=base_pred,
            sensitivity_score=sensitivity
        )

    def analyze_all_features(self, model: Any,
                             X: pd.DataFrame,
                             features: List[str] = None,
                             variation_pct: float = 0.10) -> Dict[str, SensitivityResult]:
        """
        Analyze sensitivity for all features.
        """
        if features is None:
            features = X.columns.tolist()
        
        results = {}
        for feature in features:
            if feature in X.columns:
                results[feature] = self.compute_sensitivity(
                    model, X, feature, variation_pct
                )
        
        return results
    
    def flag_high_sensitivity(self, results: Dict[str, SensitivityResult]) -> List[str]:
        """
        Flag features with high sensitivity.
        
        Returns:
            List of feature names with sensitivity > threshold
        """
        high_sens = []
        
        for feature, result in results.items():
            if result.sensitivity_score > self.high_sensitivity_threshold:
                high_sens.append(feature)
        
        return high_sens
    
    def generate_scenario_variations(self, base_value: float,
                                     variation_pct: float) -> Tuple[float, float]:
        """
        Generate scenario variations.
        
        Args:
            base_value: Base value
            variation_pct: Variation percentage
            
        Returns:
            Tuple of (lower_value, upper_value)
        """
        lower = base_value * (1 - variation_pct)
        upper = base_value * (1 + variation_pct)
        return lower, upper
    
    def compute_prediction_impact(self, base_pred: float,
                                  varied_pred: float) -> float:
        """
        Compute percentage impact on prediction.
        """
        if abs(base_pred) < 1e-6:
            return 0.0
        return (varied_pred - base_pred) / base_pred * 100
