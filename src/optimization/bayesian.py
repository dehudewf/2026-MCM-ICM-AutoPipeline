"""
Bayesian Uncertainty Quantification Module
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class UncertaintyResult:
    """Result of uncertainty quantification"""
    point_estimate: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence_level: float
    high_uncertainty_indices: list


class BayesianUncertainty:
    """
    Quantifies prediction uncertainty using Bayesian methods.
    
    Provides confidence intervals for predictions.
    """
    
    def __init__(self, n_samples: int = 2000, n_tune: int = 1000,
                 confidence_level: float = 0.95,
                 wide_ci_threshold: float = 0.20):
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.confidence_level = confidence_level
        self.wide_ci_threshold = wide_ci_threshold
        self.trace = None
    
    def compute_confidence_intervals(self, 
                                     predictions: Dict[str, np.ndarray],
                                     confidence: float = None) -> UncertaintyResult:
        """
        Compute confidence intervals from ensemble predictions.
        
        Uses prediction variance across models as uncertainty estimate.
        """
        if confidence is None:
            confidence = self.confidence_level
        
        # Stack predictions
        pred_array = np.array(list(predictions.values()))
        
        # Point estimate (mean)
        point_estimate = np.mean(pred_array, axis=0)
        
        # Compute percentiles
        alpha = 1 - confidence
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(pred_array, lower_percentile, axis=0)
        upper_bound = np.percentile(pred_array, upper_percentile, axis=0)
        
        # Find high uncertainty predictions
        ci_width = upper_bound - lower_bound
        relative_width = ci_width / np.abs(point_estimate + 1e-6)
        high_uncertainty = np.where(relative_width > self.wide_ci_threshold)[0].tolist()
        
        return UncertaintyResult(
            point_estimate=point_estimate,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence,
            high_uncertainty_indices=high_uncertainty
        )

    def compute_percentiles(self, samples: np.ndarray,
                            lower_pct: float = 2.5,
                            upper_pct: float = 97.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute percentile bounds from samples.
        """
        lower = np.percentile(samples, lower_pct, axis=0)
        upper = np.percentile(samples, upper_pct, axis=0)
        return lower, upper
    
    def flag_wide_intervals(self, point_estimate: np.ndarray,
                            lower: np.ndarray,
                            upper: np.ndarray) -> list:
        """
        Flag predictions with wide confidence intervals.
        
        Returns indices where CI width > threshold * point estimate
        """
        ci_width = upper - lower
        relative_width = ci_width / (np.abs(point_estimate) + 1e-6)
        
        return np.where(relative_width > self.wide_ci_threshold)[0].tolist()
    
    def validate_interval_coverage(self, point_estimate: np.ndarray,
                                   lower: np.ndarray,
                                   upper: np.ndarray) -> bool:
        """
        Validate that point estimates are within intervals.
        """
        return np.all((point_estimate >= lower) & (point_estimate <= upper))
