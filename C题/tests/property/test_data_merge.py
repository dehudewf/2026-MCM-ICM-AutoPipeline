"""
Property-based tests for data merging
Feature: olympic-medal-prediction, Property 4: Merge Preservation
"""
import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data.merger import DataMerger


@st.composite
def medal_and_host_dataframes(draw):
    """Generate matching medal and host DataFrames"""
    # Generate years that appear in both datasets
    years = draw(st.lists(
        st.sampled_from([1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]),
        min_size=2, max_size=6, unique=True
    ))
    
    # Generate medals data
    medal_rows = []
    countries = ['USA', 'CHN', 'GBR', 'GER', 'FRA']
    for year in years:
        for country in countries:
            medal_rows.append({
                'year': year,
                'country': country,
                'gold': draw(st.integers(min_value=0, max_value=50)),
                'silver': draw(st.integers(min_value=0, max_value=50)),
                'bronze': draw(st.integers(min_value=0, max_value=50)),
                'total': 0  # Will be calculated
            })
    
    medals_df = pd.DataFrame(medal_rows)
    medals_df['total'] = medals_df['gold'] + medals_df['silver'] + medals_df['bronze']
    
    # Generate hosts data
    host_countries = draw(st.lists(
        st.sampled_from(['USA', 'CHN', 'GBR', 'GER', 'FRA', 'JPN', 'AUS', 'BRA']),
        min_size=len(years), max_size=len(years)
    ))
    hosts_df = pd.DataFrame({
        'year': years,
        'host_country': host_countries,
        'host_city': ['City' + str(i) for i in range(len(years))]
    })
    
    return medals_df, hosts_df


class TestMergePreservation:
    """
    Feature: olympic-medal-prediction, Property 4: Merge Preservation
    
    For any set of dataframes with common keys (year, country), the merged dataset
    should preserve all matching records and the number of rows should equal the
    number of unique key combinations present in all input dataframes.
    """
    
    @given(data=medal_and_host_dataframes())
    @settings(max_examples=100)
    def test_merge_preserves_medal_rows(self, data):
        """Merge should preserve all rows from medals DataFrame"""
        medals_df, hosts_df = data
        
        merger = DataMerger()
        result = merger.merge_datasets(medals_df, hosts_df)
        
        assert result.success == True
        # All medal rows should be preserved (left join)
        assert result.rows_after == len(medals_df)
    
    @given(data=medal_and_host_dataframes())
    @settings(max_examples=100)
    def test_merge_adds_host_columns(self, data):
        """Merge should add host columns to result"""
        medals_df, hosts_df = data
        
        merger = DataMerger()
        result = merger.merge_datasets(medals_df, hosts_df)
        
        assert result.success == True
        assert 'host_country' in result.data.columns
        assert 'host_city' in result.data.columns
        assert 'is_host' in result.data.columns
    
    @given(data=medal_and_host_dataframes())
    @settings(max_examples=100)
    def test_host_indicator_correctness(self, data):
        """is_host should be 1 when country == host_country"""
        medals_df, hosts_df = data
        
        merger = DataMerger()
        result = merger.merge_datasets(medals_df, hosts_df)
        
        assert result.success == True
        df = result.data
        
        # Check is_host indicator
        for _, row in df.iterrows():
            expected_is_host = 1 if row['country'] == row['host_country'] else 0
            assert row['is_host'] == expected_is_host

    @given(data=medal_and_host_dataframes())
    @settings(max_examples=50)
    def test_merge_tracks_unmatched(self, data):
        """Merge should track unmatched keys"""
        medals_df, hosts_df = data
        
        merger = DataMerger()
        result = merger.merge_datasets(medals_df, hosts_df)
        
        assert result.success == True
        # unmatched_keys should be a dict
        assert isinstance(result.unmatched_keys, dict)
    
    @given(data=medal_and_host_dataframes())
    @settings(max_examples=50)
    def test_original_columns_preserved(self, data):
        """All original medal columns should be preserved"""
        medals_df, hosts_df = data
        
        merger = DataMerger()
        result = merger.merge_datasets(medals_df, hosts_df)
        
        assert result.success == True
        
        # All original medal columns should exist
        for col in medals_df.columns:
            assert col in result.data.columns
    
    @given(data=medal_and_host_dataframes())
    @settings(max_examples=50)
    def test_merge_values_unchanged(self, data):
        """Original medal values should not be changed by merge"""
        medals_df, hosts_df = data
        
        merger = DataMerger()
        result = merger.merge_datasets(medals_df, hosts_df)
        
        assert result.success == True
        
        # Check that medal values are unchanged
        merged = result.data
        for idx, row in medals_df.iterrows():
            # Find matching row in merged
            match = merged[
                (merged['year'] == row['year']) & 
                (merged['country'] == row['country'])
            ]
            assert len(match) >= 1
            assert match.iloc[0]['gold'] == row['gold']
            assert match.iloc[0]['silver'] == row['silver']
            assert match.iloc[0]['bronze'] == row['bronze']
