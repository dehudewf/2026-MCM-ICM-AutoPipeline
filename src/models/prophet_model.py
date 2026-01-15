"""
Prophet Time Series Model
Implements Facebook Prophet for Olympic medal prediction
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    from prophet import Prophet
except ImportError:
    Prophet = None


@dataclass
class ProphetResult:
    """Result of Prophet prediction"""
    predictions: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    components: Optional[Dict[str, np.ndarray]] = None


class ProphetModel:
    """
    Prophet model for time series forecasting.
    
    Features:
    - 4-year Olympic seasonality
    - External regressors (GDP, population)
    - Trend changepoint detection
    """
    
    def __init__(self, 
                 yearly_seasonality: bool = True,
                 changepoint_prior_scale: float = 0.05,
                 seasonality_mode: str = 'additive'):
        """
        Initialize Prophet model.
        """
        self.yearly_seasonality = yearly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_mode = seasonality_mode
        self.model = None
        self.regressors = []
        self.is_fitted = False
    
    def add_regressor(self, name: str) -> None:
        """Add external regressor"""
        self.regressors.append(name)

    def fit(self, df: pd.DataFrame) -> bool:
        """
        Fit Prophet model.
        
        Args:
            df: DataFrame with 'ds' (date) and 'y' (target) columns
        """
        if Prophet is None:
            return False
        
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_mode=self.seasonality_mode
        )
        
        # Add regressors
        for reg in self.regressors:
            if reg in df.columns:
                self.model.add_regressor(reg)
        
        self.model.fit(df)
        self.is_fitted = True
        return True
    
    def predict(self, future_df: pd.DataFrame) -> ProphetResult:
        """
        Make predictions.
        
        Args:
            future_df: DataFrame with future dates and regressor values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecast = self.model.predict(future_df)
        
        return ProphetResult(
            predictions=forecast['yhat'].values,
            lower_bound=forecast['yhat_lower'].values,
            upper_bound=forecast['yhat_upper'].values,
            components={
                'trend': forecast['trend'].values if 'trend' in forecast else None
            }
        )
    
    def make_future_dataframe(self, periods: int = 1, 
                               freq: str = '4Y') -> pd.DataFrame:
        """Create future dataframe for prediction"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.make_future_dataframe(periods=periods, freq=freq)
