"""
Event Change Adjustment Module
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class EventChangeAdjuster:
    """
    Adjusts predictions for Olympic event additions/removals.
    """
    
    def __init__(self):
        self.country_event_rates = {}
    
    def calculate_event_rates(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate medal-per-event rates for each country.
        
        Args:
            df: DataFrame with 'country', 'total', 'events_participated'
        """
        rates = {}
        
        for country in df['country'].unique():
            country_data = df[df['country'] == country]
            if 'events_participated' in country_data.columns:
                total_medals = country_data['total'].sum()
                total_events = country_data['events_participated'].sum()
                if total_events > 0:
                    rates[country] = total_medals / total_events
        
        self.country_event_rates = rates
        return rates
    
    def estimate_event_impact(self, country: str,
                              new_events: List[str] = None,
                              removed_events: List[str] = None) -> float:
        """
        Estimate medal impact of event changes.
        
        Returns:
            Net medal change (positive for additions, negative for removals)
        """
        rate = self.country_event_rates.get(country, 0.1)  # Default rate
        
        n_new = len(new_events) if new_events else 0
        n_removed = len(removed_events) if removed_events else 0
        
        return rate * (n_new - n_removed)
    
    def apply_adjustment(self, predictions: np.ndarray,
                         adjustments: Dict[int, float]) -> np.ndarray:
        """
        Apply event change adjustments.
        
        Args:
            predictions: Base predictions
            adjustments: Dict mapping index to adjustment value
        """
        adjusted = predictions.copy()
        
        for idx, adj in adjustments.items():
            if 0 <= idx < len(adjusted):
                adjusted[idx] = max(0, adjusted[idx] + adj)
        
        return adjusted
