"""
Host Country Effect Adjustment Module
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HostEffectResult:
    """Result of host effect calculation"""
    average_effect: float
    effect_by_country: Dict[str, float]
    sample_size: int


class HostEffectAdjuster:
    """
    Adjusts predictions for host country advantage.
    
    Historical data shows host countries gain 15-25% more medals.
    """
    
    def __init__(self, start_year: int = 2000, end_year: int = 2024):
        self.start_year = start_year
        self.end_year = end_year
        self.historical_effect = None
    
    def calculate_historical_effect(self, df: pd.DataFrame) -> HostEffectResult:
        """
        Calculate average host country effect from historical data.
        
        Args:
            df: DataFrame with 'year', 'country', 'total', 'is_host', 'total_lag1'
        """
        # Filter to analysis period
        data = df[(df['year'] >= self.start_year) & (df['year'] <= self.end_year)]
        
        # Get host country rows
        host_data = data[data['is_host'] == 1].copy()
        
        if len(host_data) == 0:
            return HostEffectResult(0.0, {}, 0)
        
        # Calculate effect as ratio of current to previous medals
        host_data['effect'] = host_data['total'] / host_data['total_lag1'] - 1
        
        # Remove invalid values
        host_data = host_data[host_data['effect'].notna() & np.isfinite(host_data['effect'])]
        
        avg_effect = host_data['effect'].mean()
        effect_by_country = host_data.groupby('country')['effect'].mean().to_dict()
        
        self.historical_effect = avg_effect
        
        return HostEffectResult(
            average_effect=avg_effect,
            effect_by_country=effect_by_country,
            sample_size=len(host_data)
        )

    def apply_adjustment(self, predictions: np.ndarray,
                         host_countries: List[str],
                         effect: float = None) -> np.ndarray:
        """
        Apply host effect adjustment to predictions.
        
        Args:
            predictions: Base predictions
            host_countries: List indicating which predictions are for host
            effect: Effect rate (default: use historical)
        """
        if effect is None:
            effect = self.historical_effect or 0.185  # Default 18.5%
        
        adjusted = predictions.copy()
        
        for i, is_host in enumerate(host_countries):
            if is_host:
                adjusted[i] = predictions[i] * (1 + effect)
        
        return adjusted
    
    def apply_multiplicative_adjustment(self, baseline: float, 
                                        effect_rate: float) -> float:
        """
        Apply multiplicative adjustment.
        
        Args:
            baseline: Base prediction
            effect_rate: Host effect rate (e.g., 0.185 for 18.5%)
            
        Returns:
            Adjusted prediction
        """
        return baseline * (1 + effect_rate)
    
    def validate_effect_range(self, effect: float,
                              expected_range: Tuple[float, float] = (0.15, 0.25)) -> bool:
        """Check if effect is within expected range"""
        return expected_range[0] <= effect <= expected_range[1]
