"""
Property-based tests for data validation
Feature: olympic-medal-prediction, Property 1: Data Integrity Validation
"""
import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data.cleaner import DataCleaner, ValidationReport


# Strategies for generating test data
@st.composite
def valid_medal_dataframe(draw):
    """Generate a valid medal DataFrame"""
    n_rows = draw(st.integers(min_value=1, max_value=50))
    years = draw(st.lists(
        st.integers(min_value=1896, max_value=2024),
        min_size=n_rows, max_size=n_rows
    ))
    countries = draw(st.lists(
        st.sampled_from(['USA', 'CHN', 'GBR', 'GER', 'FRA', 'JPN', 'AUS']),
        min_size=n_rows, max_size=n_rows
    ))
    gold = draw(st.lists(
        st.integers(min_value=0, max_value=50),
        min_size=n_rows, max_size=n_rows
    ))
    silver = draw(st.lists(
        st.integers(min_value=0, max_value=50),
        min_size=n_rows, max_size=n_rows
    ))
    bronze = draw(st.lists(
        st.integers(min_value=0, max_value=50),
        min_size=n_rows, max_size=n_rows
    ))
    total = [g + s + b for g, s, b in zip(gold, silver, bronze)]
    
    return pd.DataFrame({
        'year': years,
        'country': countries,
        'gold': gold,
        'silver': silver,
        'bronze': bronze,
        'total': total
    })


@st.composite
def dataframe_with_missing_values(draw):
    """Generate DataFrame with some missing values"""
    df = draw(valid_medal_dataframe())
    # Introduce some missing values
    n_missing = draw(st.integers(min_value=1, max_value=min(5, len(df))))
    for _ in range(n_missing):
        row_idx = draw(st.integers(min_value=0, max_value=len(df)-1))
        col = draw(st.sampled_from(['gold', 'silver', 'bronze']))
        df.loc[row_idx, col] = np.nan
    return df


@st.composite
def dataframe_with_invalid_values(draw):
    """Generate DataFrame with invalid values (negative medals)"""
    df = draw(valid_medal_dataframe())
    assume(len(df) > 0)
    # Introduce negative values
    row_idx = draw(st.integers(min_value=0, max_value=len(df)-1))
    col = draw(st.sampled_from(['gold', 'silver', 'bronze']))
    df.loc[row_idx, col] = -1
    return df, col


class TestDataIntegrityValidation:
    """
    Feature: olympic-medal-prediction, Property 1: Data Integrity Validation
    
    For any loaded dataset with integrity issues (missing values, type mismatches,
    invalid ranges), the validation system should detect and report all issues
    without false negatives.
    """
    
    @given(df=valid_medal_dataframe())
    @settings(max_examples=100)
    def test_valid_data_passes_validation(self, df):
        """Valid data should pass validation without errors"""
        cleaner = DataCleaner()
        report = cleaner.validate_data_integrity(df, dataset_type='medals')
        
        # Valid data should be marked as valid
        assert report.is_valid == True
        assert report.total_rows == len(df)
        assert report.total_columns == len(df.columns)

    @given(df=dataframe_with_missing_values())
    @settings(max_examples=100)
    def test_missing_values_detected(self, df):
        """Missing values should be detected and reported"""
        cleaner = DataCleaner()
        report = cleaner.validate_data_integrity(df, dataset_type='medals')
        
        # Count actual missing values
        actual_missing = df.isna().sum().sum()
        
        if actual_missing > 0:
            # Should have missing value issues reported
            missing_issues = [i for i in report.issues if i.issue_type == 'missing_values']
            assert len(missing_issues) > 0
            
            # Total reported missing should match actual
            reported_missing = sum(report.missing_values.values())
            assert reported_missing == actual_missing
    
    @given(data=dataframe_with_invalid_values())
    @settings(max_examples=100)
    def test_negative_values_detected(self, data):
        """Negative medal values should be detected as errors"""
        df, invalid_col = data
        cleaner = DataCleaner()
        report = cleaner.validate_data_integrity(df, dataset_type='medals')
        
        # Should detect invalid range
        range_issues = [i for i in report.issues 
                       if i.issue_type == 'invalid_range' and i.column == invalid_col]
        assert len(range_issues) > 0
        
        # Should mark as invalid due to error
        assert report.is_valid == False
    
    @given(df=valid_medal_dataframe())
    @settings(max_examples=50)
    def test_missing_columns_detected(self, df):
        """Missing required columns should be detected"""
        # Remove a required column
        df_missing = df.drop(columns=['gold'])
        
        cleaner = DataCleaner()
        report = cleaner.validate_data_integrity(df_missing, dataset_type='medals')
        
        # Should detect missing column
        missing_col_issues = [i for i in report.issues if i.issue_type == 'missing_column']
        assert len(missing_col_issues) > 0
        assert report.is_valid == False
