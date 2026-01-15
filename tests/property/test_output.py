"""
Property-based tests for output module
Feature: olympic-medal-prediction, Properties 28-29
"""
import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from hypothesis import given, strategies as st, settings

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from output.reporter import PredictionReporter


class TestPredictionOutputCompleteness:
    """
    Feature: olympic-medal-prediction, Property 28: Prediction Output Completeness
    
    For any prediction run, the output table should contain all required columns
    for all countries.
    """
    
    @given(
        n_countries=st.integers(min_value=5, max_value=50),
        predictions=st.lists(
            st.floats(min_value=10, max_value=150, allow_nan=False),
            min_size=5, max_size=50
        )
    )
    @settings(max_examples=50)
    def test_all_required_columns_present(self, n_countries, predictions):
        """Output table should have all required columns"""
        # Create predictions dict
        n = min(n_countries, len(predictions))
        pred_dict = {f'C{i}': predictions[i] for i in range(n)}
        
        reporter = PredictionReporter()
        df = reporter.create_prediction_table(pred_dict)
        
        # Check all required columns
        assert reporter.validate_table_columns(df)
        
        # Check all countries present
        assert len(df) == n
    
    @given(
        predictions=st.dictionaries(
            st.sampled_from(['USA', 'CHN', 'GBR', 'GER', 'FRA', 'JPN']),
            st.floats(min_value=10, max_value=150, allow_nan=False),
            min_size=3, max_size=6
        )
    )
    @settings(max_examples=50)
    def test_rank_ordering(self, predictions):
        """Ranks should be ordered by total predicted medals"""
        reporter = PredictionReporter()
        df = reporter.create_prediction_table(predictions)
        
        # Ranks should be 1, 2, 3, ...
        expected_ranks = list(range(1, len(df) + 1))
        assert df['rank'].tolist() == expected_ranks
        
        # Higher ranks should have more medals
        for i in range(len(df) - 1):
            assert df.iloc[i]['total_predicted'] >= df.iloc[i+1]['total_predicted']


class TestFileExportFormatCorrectness:
    """
    Feature: olympic-medal-prediction, Property 29: File Export Format Correctness
    
    For any exported file, the format should be correct (CSV for data, PNG for images).
    """
    
    def test_csv_export_format(self):
        """CSV export should create valid CSV file"""
        predictions = {'USA': 140, 'CHN': 90, 'GBR': 65}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = PredictionReporter(output_dir=tmpdir)
            df = reporter.create_prediction_table(predictions)
            filepath = reporter.save_to_csv(df, 'test.csv')
            
            # File should exist
            assert os.path.exists(filepath)
            
            # Should be readable as CSV
            loaded = pd.read_csv(filepath)
            assert len(loaded) == len(predictions)
            assert reporter.validate_table_columns(loaded)
