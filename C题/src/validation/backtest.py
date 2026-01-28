"""
Backtesting and Validation Module
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class BacktestResult:
    """Result of backtesting"""
    train_years: List[int]
    test_year: int
    predictions: np.ndarray
    actuals: np.ndarray
    metrics: Dict[str, float]


class BacktestValidator:
    """
    Validates models using time series cross-validation and backtesting.
    
    Ensures no future data leakage in training.
    """
    
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
    
    def time_series_split(self, df: pd.DataFrame,
                          year_col: str = 'year') -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create time series cross-validation splits.
        
        Args:
            df: DataFrame with year column
            year_col: Name of year column
            
        Returns:
            List of (train_df, test_df) tuples
        """
        years = sorted(df[year_col].unique())
        n_years = len(years)
        
        if n_years < self.n_splits + 1:
            raise ValueError(f"Not enough years for {self.n_splits} splits")
        
        splits = []
        min_train_size = n_years // 2
        
        for i in range(self.n_splits):
            # Calculate split point
            test_idx = min_train_size + i * ((n_years - min_train_size) // self.n_splits)
            test_year = years[test_idx]
            
            train_df = df[df[year_col] < test_year]
            test_df = df[df[year_col] == test_year]
            
            if len(train_df) > 0 and len(test_df) > 0:
                splits.append((train_df, test_df))
        
        return splits

    def create_temporal_split(self, df: pd.DataFrame,
                               test_year: int,
                               year_col: str = 'year') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split ensuring no future data leakage.
        
        Args:
            df: Full dataset
            test_year: Year to use as test set
            year_col: Name of year column
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train_df = df[df[year_col] < test_year].copy()
        test_df = df[df[year_col] == test_year].copy()
        
        return train_df, test_df
    
    def validate_no_leakage(self, train_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            year_col: str = 'year') -> bool:
        """
        Validate that training data doesn't contain future information.
        
        Returns:
            True if no leakage, False otherwise
        """
        if len(train_df) == 0 or len(test_df) == 0:
            return True
        
        max_train_year = train_df[year_col].max()
        min_test_year = test_df[year_col].min()
        
        return max_train_year < min_test_year
    
    def aggregate_cv_metrics(self, fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate metrics across cross-validation folds.
        
        Args:
            fold_metrics: List of metric dictionaries from each fold
            
        Returns:
            Dictionary with average metrics
        """
        if not fold_metrics:
            return {}
        
        keys = fold_metrics[0].keys()
        aggregated = {}
        
        for key in keys:
            values = [m[key] for m in fold_metrics if key in m]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
        
        return aggregated
    
    def backtest_on_year(self, df: pd.DataFrame,
                         model: Any,
                         test_year: int,
                         features: List[str],
                         target: str,
                         year_col: str = 'year') -> BacktestResult:
        """
        Backtest model on a specific year.
        """
        train_df, test_df = self.create_temporal_split(df, test_year, year_col)
        
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError(f"Invalid split for year {test_year}")
        
        # Train model
        X_train = train_df[features]
        y_train = train_df[target]
        model.fit(X_train, y_train)
        
        # Predict
        X_test = test_df[features]
        predictions = model.predict(X_test)
        actuals = test_df[target].values
        
        # Calculate metrics
        from validation.evaluator import ModelEvaluator
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(actuals, predictions).to_dict()
        
        return BacktestResult(
            train_years=train_df[year_col].unique().tolist(),
            test_year=test_year,
            predictions=predictions,
            actuals=actuals,
            metrics=metrics
        )
