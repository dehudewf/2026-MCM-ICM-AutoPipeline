"""
Feature Engineering Module
Creates predictive features from raw Olympic data
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple


class FeatureEngineer:
    """
    Creates features for Olympic medal prediction including:
    - Lagged features (historical medals)
    - Rolling statistics
    - Economic features
    - Interaction features
    - Host country indicators
    """
    
    def __init__(self):
        """Initialize FeatureEngineer"""
        pass
    
    # ==================== Lagged Features ====================
    
    def create_lagged_features(self, df: pd.DataFrame,
                                target_col: str = 'total',
                                lags: List[int] = [1, 2, 3],
                                group_by: str = 'country') -> pd.DataFrame:
        """
        Create lagged features (medals from previous Olympics).
        
        Args:
            df: DataFrame with medal data
            target_col: Column to create lags for
            lags: List of lag periods (1 = previous Olympics)
            group_by: Column to group by before creating lags
        """
        df = df.copy()
        
        # Sort by year within each group
        df = df.sort_values([group_by, 'year'])
        
        for lag in lags:
            col_name = f'{target_col}_lag{lag}'
            df[col_name] = df.groupby(group_by)[target_col].shift(lag)
        
        return df

    # ==================== Rolling Statistics ====================
    
    def create_rolling_features(self, df: pd.DataFrame,
                                 target_col: str = 'total',
                                 windows: List[int] = [3, 5],
                                 group_by: str = 'country') -> pd.DataFrame:
        """
        Create rolling statistics features.
        
        Args:
            df: DataFrame with medal data
            target_col: Column to calculate rolling stats for
            windows: List of window sizes
            group_by: Column to group by
        """
        df = df.copy()
        df = df.sort_values([group_by, 'year'])
        
        for window in windows:
            # Rolling mean
            ma_col = f'{target_col}_ma{window}'
            df[ma_col] = df.groupby(group_by)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Rolling standard deviation
            std_col = f'{target_col}_std{window}'
            df[std_col] = df.groupby(group_by)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        return df
    
    def create_trend_features(self, df: pd.DataFrame,
                               target_col: str = 'total',
                               group_by: str = 'country') -> pd.DataFrame:
        """
        Create trend features (growth rate, acceleration).
        """
        df = df.copy()
        df = df.sort_values([group_by, 'year'])
        
        # Growth rate (percent change)
        df[f'{target_col}_growth'] = df.groupby(group_by)[target_col].pct_change()
        
        # Acceleration (change in growth rate)
        df[f'{target_col}_acceleration'] = df.groupby(group_by)[f'{target_col}_growth'].diff()
        
        return df

    # ==================== Economic Features ====================
    
    def create_economic_features(self, df: pd.DataFrame,
                                  gdp_col: str = 'gdp',
                                  pop_col: str = 'population',
                                  group_by: str = 'country') -> pd.DataFrame:
        """
        Create economic features.
        
        Args:
            df: DataFrame with economic data
            gdp_col: GDP column name
            pop_col: Population column name
            group_by: Column to group by
        """
        df = df.copy()
        
        # GDP per capita
        if gdp_col in df.columns and pop_col in df.columns:
            # Handle zero/negative population
            df['gdp_per_capita'] = np.where(
                df[pop_col] > 0,
                df[gdp_col] / df[pop_col],
                0
            )
        
        # GDP growth rate
        if gdp_col in df.columns:
            df = df.sort_values([group_by, 'year'])
            df['gdp_growth'] = df.groupby(group_by)[gdp_col].pct_change()
        
        # Log GDP (handle zero/negative)
        if gdp_col in df.columns:
            df['gdp_log'] = np.log(df[gdp_col].clip(lower=1))
        
        return df
    
    def calculate_gdp_per_capita(self, gdp: float, population: float) -> float:
        """
        Calculate GDP per capita.
        
        Args:
            gdp: GDP value
            population: Population value
            
        Returns:
            GDP per capita (0 if population <= 0)
        """
        if population <= 0:
            return 0.0
        return gdp / population

    # ==================== Interaction Features ====================
    
    def create_interaction_features(self, df: pd.DataFrame,
                                     pairs: List[Tuple[str, str]] = None) -> pd.DataFrame:
        """
        Create interaction features (cross-products).
        
        Args:
            df: DataFrame with features
            pairs: List of (col1, col2) tuples to create interactions for
        """
        df = df.copy()
        
        if pairs is None:
            # Default interactions
            pairs = [
                ('gdp', 'population'),
                ('sports_budget', 'events_participated')
            ]
        
        for col1, col2 in pairs:
            if col1 in df.columns and col2 in df.columns:
                interaction_name = f'{col1}_{col2}_interaction'
                df[interaction_name] = df[col1] * df[col2]
        
        return df
    
    def calculate_interaction(self, value1: float, value2: float) -> float:
        """
        Calculate interaction (product) of two values.
        
        Args:
            value1: First value
            value2: Second value
            
        Returns:
            Product of the two values
        """
        return value1 * value2

    # ==================== Host Country Features ====================
    
    def create_host_features(self, df: pd.DataFrame,
                              country_col: str = 'country',
                              host_col: str = 'host_country') -> pd.DataFrame:
        """
        Create host country indicator features.
        
        Args:
            df: DataFrame with country and host information
            country_col: Column with country codes
            host_col: Column with host country code
        """
        df = df.copy()
        
        if country_col in df.columns and host_col in df.columns:
            # Binary host indicator
            df['is_host'] = (df[country_col] == df[host_col]).astype(int)
            
            # Host effect interaction with previous medals
            if 'total_lag1' in df.columns:
                df['host_effect'] = df['is_host'] * df['total_lag1']
        
        return df
    
    def is_host_country(self, country: str, host: str) -> int:
        """
        Check if country is the host.
        
        Args:
            country: Country code
            host: Host country code
            
        Returns:
            1 if country is host, 0 otherwise
        """
        return 1 if country == host else 0
    
    # ==================== All Features ====================
    
    def create_all_features(self, df: pd.DataFrame,
                            target_col: str = 'total',
                            group_by: str = 'country') -> pd.DataFrame:
        """
        Create all features in one call.
        """
        df = self.create_lagged_features(df, target_col, group_by=group_by)
        df = self.create_rolling_features(df, target_col, group_by=group_by)
        df = self.create_trend_features(df, target_col, group_by=group_by)
        
        if 'gdp' in df.columns:
            df = self.create_economic_features(df, group_by=group_by)
        
        df = self.create_interaction_features(df)
        df = self.create_host_features(df)
        
        return df
