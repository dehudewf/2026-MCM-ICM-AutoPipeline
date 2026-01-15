"""
ARIMA Time Series Model
Implements ARIMA for Olympic medal prediction
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    ARIMA = None
    adfuller = None


@dataclass
class StationarityResult:
    """Result of stationarity test"""
    is_stationary: bool
    adf_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    n_lags: int


@dataclass
class ARIMAResult:
    """Result of ARIMA model fitting"""
    order: Tuple[int, int, int]
    aic: float
    bic: float
    fitted: bool
    error_message: Optional[str] = None


class ARIMAModel:
    """
    ARIMA model for time series forecasting.
    
    Features:
    - Stationarity testing (ADF test)
    - Automatic differencing
    - Grid search for optimal parameters
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        """
        Initialize ARIMA model.
        
        Args:
            order: (p, d, q) order of the ARIMA model
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        self.is_fitted = False

    def test_stationarity(self, data: pd.Series, 
                          significance: float = 0.05) -> StationarityResult:
        """
        Test stationarity using Augmented Dickey-Fuller test.
        
        Args:
            data: Time series data
            significance: Significance level for test
            
        Returns:
            StationarityResult with test statistics
        """
        if adfuller is None:
            raise ImportError("statsmodels is required for stationarity testing")
        
        # Run ADF test
        result = adfuller(data.dropna(), autolag='AIC')
        
        adf_stat = result[0]
        p_value = result[1]
        n_lags = result[2]
        critical_values = result[4]
        
        # Determine if stationary
        is_stationary = p_value < significance
        
        return StationarityResult(
            is_stationary=is_stationary,
            adf_statistic=adf_stat,
            p_value=p_value,
            critical_values=critical_values,
            n_lags=n_lags
        )
    
    def difference_series(self, data: pd.Series, d: int = 1) -> pd.Series:
        """
        Difference the series d times.
        
        Args:
            data: Time series data
            d: Number of differences
            
        Returns:
            Differenced series
        """
        result = data.copy()
        for _ in range(d):
            result = result.diff()
        return result.dropna()
    
    def auto_difference(self, data: pd.Series, 
                        max_d: int = 2,
                        significance: float = 0.05) -> Tuple[pd.Series, int]:
        """
        Automatically difference until stationary.
        
        Returns:
            Tuple of (differenced series, number of differences)
        """
        d = 0
        current = data.copy()
        
        while d < max_d:
            result = self.test_stationarity(current, significance)
            if result.is_stationary:
                break
            current = self.difference_series(current, 1)
            d += 1
        
        return current, d

    def grid_search(self, data: pd.Series,
                    p_range: Tuple[int, int] = (0, 5),
                    d_range: Tuple[int, int] = (0, 2),
                    q_range: Tuple[int, int] = (0, 5),
                    criterion: str = 'aic') -> Tuple[Tuple[int, int, int], float]:
        """
        Grid search for optimal ARIMA parameters.
        
        Args:
            data: Time series data
            p_range: Range for p parameter (min, max)
            d_range: Range for d parameter (min, max)
            q_range: Range for q parameter (min, max)
            criterion: 'aic' or 'bic'
            
        Returns:
            Tuple of (best_order, best_score)
        """
        if ARIMA is None:
            raise ImportError("statsmodels is required for ARIMA")
        
        best_score = float('inf')
        best_order = (1, 1, 1)
        
        for p in range(p_range[0], p_range[1] + 1):
            for d in range(d_range[0], d_range[1] + 1):
                for q in range(q_range[0], q_range[1] + 1):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = ARIMA(data, order=(p, d, q))
                            fitted = model.fit()
                            
                            score = fitted.aic if criterion == 'aic' else fitted.bic
                            
                            if score < best_score:
                                best_score = score
                                best_order = (p, d, q)
                    except Exception:
                        continue
        
        self.order = best_order
        return best_order, best_score

    def fit(self, data: pd.Series) -> ARIMAResult:
        """
        Fit ARIMA model to data.
        
        Args:
            data: Time series data
            
        Returns:
            ARIMAResult with fitting information
        """
        if ARIMA is None:
            return ARIMAResult(
                order=self.order,
                aic=0, bic=0, fitted=False,
                error_message="statsmodels not installed"
            )
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = ARIMA(data, order=self.order)
                self.fitted_model = self.model.fit()
                self.is_fitted = True
                
                return ARIMAResult(
                    order=self.order,
                    aic=self.fitted_model.aic,
                    bic=self.fitted_model.bic,
                    fitted=True
                )
        except Exception as e:
            self.is_fitted = False
            return ARIMAResult(
                order=self.order,
                aic=0, bic=0, fitted=False,
                error_message=str(e)
            )
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Predict future values.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return np.array(forecast)
    
    def get_aic_bic(self) -> Tuple[float, float]:
        """Get AIC and BIC of fitted model"""
        if not self.is_fitted:
            return (0.0, 0.0)
        return (self.fitted_model.aic, self.fitted_model.bic)
