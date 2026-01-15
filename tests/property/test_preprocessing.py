"""
Property-based tests for data preprocessing
Feature: olympic-medal-prediction, Property 2 & 3: Imputation and Outlier Detection
"""
import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from scipy import stats

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data.preprocessor import DataPreprocessor


@st.composite
def dataframe_with_missing(draw):
    """Generate DataFrame with missing values in numeric columns"""
    n_rows = draw(st.integers(min_value=10, max_value=50))
    
    # Generate base data
    data = {
        'year': list(range(1900, 1900 + n_rows)),
        'country': ['USA'] * n_rows,
        'gold': draw(st.lists(
            st.integers(min_value=0, max_value=50),
            min_size=n_rows, max_size=n_rows
        )),
        'silver': draw(st.lists(
            st.integers(min_value=0, max_value=50),
            min_size=n_rows, max_size=n_rows
        ))
    }
    
    df = pd.DataFrame(data)
    df['gold'] = df['gold'].astype(float)
    df['silver'] = df['silver'].astype(float)
    
    # Introduce missing values (not at edges for interpolation)
    n_missing = draw(st.integers(min_value=1, max_value=min(5, n_rows - 2)))
    for _ in range(n_missing):
        row_idx = draw(st.integers(min_value=1, max_value=n_rows - 2))
        col = draw(st.sampled_from(['gold', 'silver']))
        df.loc[row_idx, col] = np.nan
    
    return df


class TestImputationCompleteness:
    """
    Feature: olympic-medal-prediction, Property 2: Imputation Completeness
    
    For any dataset with missing values, after applying imputation strategies,
    the resulting dataset should contain zero missing values in all required columns.
    """
    
    @given(df=dataframe_with_missing())
    @settings(max_examples=100)
    def test_interpolation_removes_all_missing(self, df):
        """Interpolation should remove all missing values"""
        preprocessor = DataPreprocessor()
        
        # Verify there are missing values before
        assert df.isna().sum().sum() > 0
        
        # Apply interpolation
        result = preprocessor.impute_missing_interpolate(
            df, columns=['gold', 'silver']
        )
        
        # Should have no missing values after
        assert result['gold'].isna().sum() == 0
        assert result['silver'].isna().sum() == 0
    
    @given(df=dataframe_with_missing())
    @settings(max_examples=100)
    def test_knn_removes_all_missing(self, df):
        """KNN imputation should remove all missing values"""
        preprocessor = DataPreprocessor()
        
        # Verify there are missing values before
        assert df.isna().sum().sum() > 0
        
        # Apply KNN imputation
        result = preprocessor.impute_missing_knn(
            df, columns=['gold', 'silver'], n_neighbors=3
        )
        
        # Should have no missing values after
        assert result['gold'].isna().sum() == 0
        assert result['silver'].isna().sum() == 0
    
    @given(df=dataframe_with_missing())
    @settings(max_examples=100)
    def test_mean_removes_all_missing(self, df):
        """Mean imputation should remove all missing values"""
        preprocessor = DataPreprocessor()
        
        # Verify there are missing values before
        assert df.isna().sum().sum() > 0
        
        # Apply mean imputation
        result = preprocessor.impute_missing_mean(
            df, columns=['gold', 'silver']
        )
        
        # Should have no missing values after
        assert result['gold'].isna().sum() == 0
        assert result['silver'].isna().sum() == 0


@st.composite
def dataframe_with_outliers(draw):
    """Generate DataFrame with known outliers"""
    n_rows = draw(st.integers(min_value=20, max_value=50))
    
    # Generate normal data
    values = draw(st.lists(
        st.floats(min_value=10.0, max_value=50.0, allow_nan=False),
        min_size=n_rows, max_size=n_rows
    ))
    
    df = pd.DataFrame({'value': values})
    
    # Add known outliers (very high values)
    n_outliers = draw(st.integers(min_value=1, max_value=3))
    outlier_indices = []
    for i in range(n_outliers):
        idx = draw(st.integers(min_value=0, max_value=n_rows - 1))
        # Make it a clear outlier (10x the max normal value)
        df.loc[idx, 'value'] = 500.0 + i * 100
        outlier_indices.append(idx)
    
    return df, outlier_indices


class TestOutlierDetectionCorrectness:
    """
    Feature: olympic-medal-prediction, Property 3: Outlier Detection Correctness
    
    For any dataset and outlier detection method (Z-score or IQR), values flagged
    as outliers should satisfy the mathematical definition of the method.
    """
    
    @given(data=dataframe_with_outliers())
    @settings(max_examples=100)
    def test_zscore_outliers_exceed_threshold(self, data):
        """Z-score detected outliers should have |z| > threshold"""
        df, known_outliers = data
        threshold = 3.0
        
        preprocessor = DataPreprocessor()
        detected = preprocessor.detect_outliers_zscore(
            df, columns=['value'], threshold=threshold
        )
        
        if 'value' in detected:
            # Verify each detected outlier has z-score > threshold
            z_scores = np.abs(stats.zscore(df['value']))
            for idx in detected['value']:
                assert z_scores[idx] > threshold

    @given(data=dataframe_with_outliers())
    @settings(max_examples=100)
    def test_iqr_outliers_outside_bounds(self, data):
        """IQR detected outliers should be outside Q1-1.5*IQR to Q3+1.5*IQR"""
        df, known_outliers = data
        multiplier = 1.5
        
        preprocessor = DataPreprocessor()
        detected = preprocessor.detect_outliers_iqr(
            df, columns=['value'], multiplier=multiplier
        )
        
        if 'value' in detected:
            Q1 = df['value'].quantile(0.25)
            Q3 = df['value'].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR
            
            # Verify each detected outlier is outside bounds
            for idx in detected['value']:
                value = df.loc[idx, 'value']
                assert value < lower or value > upper
    
    @given(data=dataframe_with_outliers())
    @settings(max_examples=50)
    def test_known_outliers_detected(self, data):
        """Known extreme outliers should be detected"""
        df, known_outliers = data
        
        preprocessor = DataPreprocessor()
        
        # Use IQR method which should catch extreme values
        detected = preprocessor.detect_outliers_iqr(
            df, columns=['value'], multiplier=1.5
        )
        
        # At least some known outliers should be detected
        if 'value' in detected:
            detected_set = set(detected['value'])
            known_set = set(known_outliers)
            # Check overlap
            overlap = detected_set.intersection(known_set)
            # Most known outliers should be detected (they are extreme)
            assert len(overlap) >= len(known_outliers) * 0.5
    
    @given(data=dataframe_with_outliers())
    @settings(max_examples=50)
    def test_capping_brings_within_bounds(self, data):
        """Capped values should be within bounds"""
        df, _ = data
        
        preprocessor = DataPreprocessor()
        capped = preprocessor.cap_outliers(df, columns=['value'], method='iqr')
        
        Q1 = df['value'].quantile(0.25)
        Q3 = df['value'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # All capped values should be within bounds
        assert capped['value'].min() >= lower
        assert capped['value'].max() <= upper
