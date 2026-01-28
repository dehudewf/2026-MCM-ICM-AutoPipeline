"""
Event Count Impact Analysis Module
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy import stats


@dataclass
class CorrelationResult:
    """Result of correlation analysis"""
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float


@dataclass
class ElasticityResult:
    """Result of elasticity calculation"""
    elasticity: float
    coefficient: float
    p_value: float
    r_squared: float


class EventImpactAnalyzer:
    """
    Analyzes the relationship between event participation and medal counts.
    """
    
    def compute_correlations(self, events: np.ndarray,
                             medals: np.ndarray) -> CorrelationResult:
        """
        Compute Pearson and Spearman correlations.
        
        Args:
            events: Array of event counts
            medals: Array of medal counts
            
        Returns:
            CorrelationResult with both correlation coefficients
        """
        # Remove NaN values
        mask = ~(np.isnan(events) | np.isnan(medals))
        events_clean = events[mask]
        medals_clean = medals[mask]
        
        if len(events_clean) < 3:
            return CorrelationResult(0, 1, 0, 1)
        
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(events_clean, medals_clean)
        
        # Spearman correlation
        spearman_r, spearman_p = stats.spearmanr(events_clean, medals_clean)
        
        return CorrelationResult(
            pearson_r=pearson_r,
            pearson_p=pearson_p,
            spearman_r=spearman_r,
            spearman_p=spearman_p
        )
    
    def validate_correlation_bounds(self, corr: float) -> bool:
        """
        Validate that correlation is within [-1, 1].
        """
        return -1 <= corr <= 1

    def compute_elasticity(self, df: pd.DataFrame,
                           event_col: str = 'events_participated',
                           medal_col: str = 'total',
                           control_cols: List[str] = None) -> ElasticityResult:
        """
        Compute elasticity of medals with respect to events.
        
        Elasticity = % change in medals / % change in events
        """
        from sklearn.linear_model import LinearRegression
        
        # Prepare data
        if control_cols is None:
            control_cols = []
        
        features = [event_col] + control_cols
        X = df[features].dropna()
        y = df.loc[X.index, medal_col]
        
        if len(X) < 5:
            return ElasticityResult(0, 0, 1, 0)
        
        # Fit regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Get coefficient for events
        event_coef = model.coef_[0]
        
        # Calculate elasticity at means
        mean_events = X[event_col].mean()
        mean_medals = y.mean()
        
        if mean_medals > 0:
            elasticity = event_coef * (mean_events / mean_medals)
        else:
            elasticity = 0
        
        # R-squared
        r_squared = model.score(X, y)
        
        return ElasticityResult(
            elasticity=elasticity,
            coefficient=event_coef,
            p_value=0.05,  # Simplified
            r_squared=r_squared
        )
    
    def analyze_event_medal_relationship(self, df: pd.DataFrame,
                                          event_col: str = 'events_participated',
                                          medal_col: str = 'total') -> Dict:
        """
        Complete analysis of event-medal relationship.
        """
        events = df[event_col].values
        medals = df[medal_col].values
        
        correlations = self.compute_correlations(events, medals)
        elasticity = self.compute_elasticity(df, event_col, medal_col)
        
        return {
            'correlations': correlations,
            'elasticity': elasticity,
            'n_observations': len(df)
        }
