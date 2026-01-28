"""
Regional Analysis Module
Analyzes medal predictions by region, with focus on South America
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass


# South American country codes
SOUTH_AMERICAN_COUNTRIES = [
    'BRA', 'ARG', 'COL', 'CHI', 'PER', 'VEN', 'ECU', 'BOL', 'PAR', 'URU', 'GUY', 'SUR'
]

# Regional sports strengths
REGIONAL_STRENGTHS = {
    'South America': ['Football', 'Volleyball', 'Judo', 'Boxing', 'Athletics'],
    'Europe': ['Cycling', 'Rowing', 'Fencing', 'Gymnastics'],
    'Asia': ['Table Tennis', 'Badminton', 'Weightlifting', 'Diving'],
    'North America': ['Swimming', 'Basketball', 'Track and Field']
}


@dataclass
class RegionalPrediction:
    """Regional prediction result"""
    region: str
    countries: List[str]
    country_predictions: Dict[str, float]
    regional_total: float
    growth_rate: Optional[float] = None


class RegionalAnalyzer:
    """
    Analyzes medal predictions by region.
    
    Features:
    - South American country identification
    - Regional sports strengths
    - Hierarchical modeling for sparse data
    - Regional aggregation
    """
    
    def __init__(self):
        self.south_american = SOUTH_AMERICAN_COUNTRIES
        self.regional_strengths = REGIONAL_STRENGTHS
    
    def identify_south_american(self, df: pd.DataFrame,
                                 country_col: str = 'country') -> pd.DataFrame:
        """
        Identify South American countries in dataset.
        """
        df = df.copy()
        df['is_south_american'] = df[country_col].isin(self.south_american)
        return df
    
    def get_south_american_data(self, df: pd.DataFrame,
                                 country_col: str = 'country') -> pd.DataFrame:
        """
        Filter to South American countries only.
        """
        return df[df[country_col].isin(self.south_american)].copy()

    def aggregate_regional_predictions(self, 
                                       predictions: Dict[str, float],
                                       region_countries: List[str]) -> float:
        """
        Aggregate country predictions to regional total.
        
        Args:
            predictions: Dict mapping country codes to predictions
            region_countries: List of country codes in region
            
        Returns:
            Sum of predictions for countries in region
        """
        total = 0.0
        for country in region_countries:
            if country in predictions:
                total += predictions[country]
        return total
    
    def compute_growth_rate(self, df: pd.DataFrame,
                            country_col: str = 'country',
                            medal_col: str = 'total',
                            year_col: str = 'year') -> Dict[str, float]:
        """
        Compute medal growth rates by country.
        """
        growth_rates = {}
        
        for country in df[country_col].unique():
            country_data = df[df[country_col] == country].sort_values(year_col)
            if len(country_data) >= 2:
                first = country_data[medal_col].iloc[0]
                last = country_data[medal_col].iloc[-1]
                n_periods = len(country_data) - 1
                if first > 0 and n_periods > 0:
                    growth_rates[country] = (last / first) ** (1 / n_periods) - 1
        
        return growth_rates
    
    def create_regional_prediction(self, 
                                   predictions: Dict[str, float],
                                   region: str = 'South America') -> RegionalPrediction:
        """
        Create regional prediction summary.
        """
        if region == 'South America':
            countries = self.south_american
        else:
            countries = list(predictions.keys())
        
        country_preds = {c: predictions.get(c, 0) for c in countries if c in predictions}
        regional_total = sum(country_preds.values())
        
        return RegionalPrediction(
            region=region,
            countries=list(country_preds.keys()),
            country_predictions=country_preds,
            regional_total=regional_total
        )
    
    def compare_regions(self, predictions: Dict[str, float],
                        regions: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Compare predictions across regions.
        """
        data = []
        for region, countries in regions.items():
            total = self.aggregate_regional_predictions(predictions, countries)
            n_countries = len([c for c in countries if c in predictions])
            data.append({
                'region': region,
                'total_medals': total,
                'n_countries': n_countries,
                'avg_per_country': total / n_countries if n_countries > 0 else 0
            })
        
        return pd.DataFrame(data).sort_values('total_medals', ascending=False)
