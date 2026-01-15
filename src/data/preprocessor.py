"""
Data Preprocessor Module
Handles missing value imputation and outlier detection/handling
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from sklearn.impute import KNNImputer
from scipy import stats


class DataPreprocessor:
    """
    Preprocesses Olympic data including:
    - Missing value imputation
    - Outlier detection and handling
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize DataPreprocessor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
    
    # ==================== Missing Value Imputation ====================
    
    def impute_missing_interpolate(self, df: pd.DataFrame, 
                                    columns: List[str] = None,
                                    group_by: str = None,
                                    method: str = 'linear') -> pd.DataFrame:
        """
        Impute missing values using interpolation (for time series data).
        
        Args:
            df: DataFrame with missing values
            columns: Columns to impute (None = all numeric)
            group_by: Column to group by before interpolating
            method: Interpolation method ('linear', 'polynomial', etc.)
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if group_by and group_by in df.columns:
            # Interpolate within each group
            for col in columns:
                if col in df.columns:
                    df[col] = df.groupby(group_by)[col].transform(
                        lambda x: x.interpolate(method=method).ffill().bfill()
                    )
        else:
            # Interpolate entire column
            for col in columns:
                if col in df.columns:
                    df[col] = df[col].interpolate(method=method).ffill().bfill()
        
        return df
    
    def impute_missing_knn(self, df: pd.DataFrame,
                           columns: List[str] = None,
                           n_neighbors: int = 5) -> pd.DataFrame:
        """
        Impute missing values using KNN (for cross-sectional data).
        
        Args:
            df: DataFrame with missing values
            columns: Columns to impute (None = all numeric)
            n_neighbors: Number of neighbors for KNN
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Only impute specified columns
        cols_to_impute = [c for c in columns if c in df.columns]
        
        if not cols_to_impute:
            return df
        
        # Create imputer
        imputer = KNNImputer(n_neighbors=n_neighbors)
        
        # Fit and transform
        df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])
        
        return df
    
    def impute_missing_mean(self, df: pd.DataFrame,
                            columns: List[str] = None,
                            group_by: str = None) -> pd.DataFrame:
        """
        Impute missing values using mean.
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in df.columns:
                if group_by and group_by in df.columns:
                    df[col] = df.groupby(group_by)[col].transform(
                        lambda x: x.fillna(x.mean())
                    )
                else:
                    df[col] = df[col].fillna(df[col].mean())
        
        return df

    # ==================== Outlier Detection ====================
    
    def detect_outliers_zscore(self, df: pd.DataFrame,
                                columns: List[str] = None,
                                threshold: float = 3.0) -> Dict[str, List[int]]:
        """
        Detect outliers using Z-score method.
        
        Args:
            df: DataFrame to check
            columns: Columns to check (None = all numeric)
            threshold: Z-score threshold (default 3.0)
            
        Returns:
            Dictionary mapping column names to list of outlier row indices
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = {}
        
        for col in columns:
            if col in df.columns:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                # Get indices where z-score exceeds threshold
                outlier_mask = z_scores > threshold
                # Map back to original indices
                valid_indices = df[col].dropna().index
                outlier_indices = valid_indices[outlier_mask].tolist()
                if outlier_indices:
                    outliers[col] = outlier_indices
        
        return outliers
    
    def detect_outliers_iqr(self, df: pd.DataFrame,
                            columns: List[str] = None,
                            multiplier: float = 1.5) -> Dict[str, List[int]]:
        """
        Detect outliers using IQR method.
        
        Args:
            df: DataFrame to check
            columns: Columns to check (None = all numeric)
            multiplier: IQR multiplier (default 1.5)
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = {}
        
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - multiplier * IQR
                upper = Q3 + multiplier * IQR
                
                outlier_mask = (df[col] < lower) | (df[col] > upper)
                outlier_indices = df[outlier_mask].index.tolist()
                if outlier_indices:
                    outliers[col] = outlier_indices
        
        return outliers

    # ==================== Outlier Handling ====================
    
    def remove_outliers(self, df: pd.DataFrame,
                        outlier_indices: Dict[str, List[int]]) -> pd.DataFrame:
        """
        Remove rows containing outliers.
        
        Args:
            df: DataFrame to clean
            outlier_indices: Dict from detect_outliers methods
        """
        df = df.copy()
        
        # Collect all unique outlier indices
        all_indices = set()
        for indices in outlier_indices.values():
            all_indices.update(indices)
        
        # Remove rows
        df = df.drop(index=list(all_indices), errors='ignore')
        
        return df
    
    def cap_outliers(self, df: pd.DataFrame,
                     columns: List[str] = None,
                     method: str = 'iqr',
                     **kwargs) -> pd.DataFrame:
        """
        Cap outliers at threshold values (winsorization).
        
        Args:
            df: DataFrame to process
            columns: Columns to cap
            method: 'iqr' or 'zscore'
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'iqr':
                multiplier = kwargs.get('multiplier', 1.5)
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - multiplier * IQR
                upper = Q3 + multiplier * IQR
            else:  # zscore
                threshold = kwargs.get('threshold', 3.0)
                mean = df[col].mean()
                std = df[col].std()
                lower = mean - threshold * std
                upper = mean + threshold * std
            
            df[col] = df[col].clip(lower=lower, upper=upper)
        
        return df
    
    def flag_outliers(self, df: pd.DataFrame,
                      outlier_indices: Dict[str, List[int]],
                      flag_column: str = 'is_outlier') -> pd.DataFrame:
        """
        Add flag column indicating outlier rows.
        """
        df = df.copy()
        
        all_indices = set()
        for indices in outlier_indices.values():
            all_indices.update(indices)
        
        df[flag_column] = df.index.isin(all_indices).astype(int)
        
        return df
