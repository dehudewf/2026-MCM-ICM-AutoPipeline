"""
Property-based tests for feature engineering
Feature: olympic-medal-prediction, Properties 5-9
"""
import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from features.engineer import FeatureEngineer


@st.composite
def medal_time_series(draw):
    """Generate medal time series for a country"""
    n_years = draw(st.integers(min_value=5, max_value=15))
    years = list(range(2000, 2000 + n_years * 4, 4))  # Olympic years
    
    medals = draw(st.lists(
        st.integers(min_value=0, max_value=100),
        min_size=n_years, max_size=n_years
    ))
    
    return pd.DataFrame({
        'year': years,
        'country': ['USA'] * n_years,
        'total': medals
    })


class TestLaggedFeatureCorrectness:
    """
    Feature: olympic-medal-prediction, Property 5: Lagged Feature Correctness
    
    For any time series and lag value k, the lagged feature at position t
    should equal the original value at position t-k.
    """
    
    @given(df=medal_time_series())
    @settings(max_examples=100)
    def test_lag1_equals_previous_value(self, df):
        """Lag-1 feature should equal previous row's value"""
        engineer = FeatureEngineer()
        result = engineer.create_lagged_features(df, 'total', lags=[1])
        
        # For each row after the first, lag1 should equal previous total
        for i in range(1, len(result)):
            expected = result.iloc[i-1]['total']
            actual = result.iloc[i]['total_lag1']
            assert actual == expected

    @given(df=medal_time_series())
    @settings(max_examples=100)
    def test_first_lag_is_nan(self, df):
        """First row's lag feature should be NaN"""
        engineer = FeatureEngineer()
        result = engineer.create_lagged_features(df, 'total', lags=[1])
        
        assert pd.isna(result.iloc[0]['total_lag1'])
    
    @given(df=medal_time_series())
    @settings(max_examples=50)
    def test_lag2_equals_two_back(self, df):
        """Lag-2 feature should equal value from 2 rows back"""
        assume(len(df) >= 3)
        
        engineer = FeatureEngineer()
        result = engineer.create_lagged_features(df, 'total', lags=[2])
        
        for i in range(2, len(result)):
            expected = result.iloc[i-2]['total']
            actual = result.iloc[i]['total_lag2']
            assert actual == expected


class TestRollingStatisticsAccuracy:
    """
    Feature: olympic-medal-prediction, Property 6: Rolling Statistics Accuracy
    
    For any sequence of values and window size w, the rolling mean at position t
    should equal the arithmetic mean of values from position t-w+1 to t.
    """
    
    @given(df=medal_time_series())
    @settings(max_examples=100)
    def test_rolling_mean_calculation(self, df):
        """Rolling mean should equal manual calculation"""
        assume(len(df) >= 3)
        
        engineer = FeatureEngineer()
        result = engineer.create_rolling_features(df, 'total', windows=[3])
        
        # Check rolling mean for rows with full window
        for i in range(2, len(result)):
            window_values = result.iloc[i-2:i+1]['total'].values
            expected_mean = np.mean(window_values)
            actual_mean = result.iloc[i]['total_ma3']
            assert abs(actual_mean - expected_mean) < 1e-6


class TestEconomicFeatureCalculation:
    """
    Feature: olympic-medal-prediction, Property 7: Economic Feature Calculation
    
    For any GDP and population values, GDP per capita should equal GDP / population.
    """
    
    @given(
        gdp=st.floats(min_value=1e9, max_value=1e13, allow_nan=False),
        population=st.integers(min_value=1000000, max_value=1500000000)
    )
    @settings(max_examples=100)
    def test_gdp_per_capita_calculation(self, gdp, population):
        """GDP per capita should equal GDP / population"""
        engineer = FeatureEngineer()
        result = engineer.calculate_gdp_per_capita(gdp, population)
        expected = gdp / population
        assert abs(result - expected) < 1e-6
    
    @given(gdp=st.floats(min_value=1e9, max_value=1e13, allow_nan=False))
    @settings(max_examples=50)
    def test_zero_population_returns_zero(self, gdp):
        """Zero population should return 0 GDP per capita"""
        engineer = FeatureEngineer()
        result = engineer.calculate_gdp_per_capita(gdp, 0)
        assert result == 0.0


class TestInteractionFeatureMultiplication:
    """
    Feature: olympic-medal-prediction, Property 8: Interaction Feature Multiplication
    
    For any two features A and B, the interaction should equal A × B.
    """
    
    @given(
        value1=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
        value2=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_interaction_equals_product(self, value1, value2):
        """Interaction should equal product of values"""
        engineer = FeatureEngineer()
        result = engineer.calculate_interaction(value1, value2)
        expected = value1 * value2
        assert abs(result - expected) < 1e-6


class TestHostIndicatorCorrectness:
    """
    Feature: olympic-medal-prediction, Property 9: Host Indicator Correctness
    
    For any country-year pair, the host indicator should be 1 if and only if
    that country hosted the Olympics in that year.
    """
    
    @given(
        country=st.sampled_from(['USA', 'CHN', 'GBR', 'GER', 'FRA']),
        host=st.sampled_from(['USA', 'CHN', 'GBR', 'GER', 'FRA'])
    )
    @settings(max_examples=100)
    def test_host_indicator_correctness(self, country, host):
        """Host indicator should be 1 when country == host"""
        engineer = FeatureEngineer()
        result = engineer.is_host_country(country, host)
        
        if country == host:
            assert result == 1
        else:
            assert result == 0
    
    @given(country=st.sampled_from(['USA', 'CHN', 'GBR', 'GER', 'FRA']))
    @settings(max_examples=50)
    def test_host_indicator_in_dataframe(self, country):
        """Host indicator in DataFrame should match logic"""
        df = pd.DataFrame({
            'year': [2024],
            'country': [country],
            'host_country': ['FRA'],
            'total': [50]
        })
        
        engineer = FeatureEngineer()
        result = engineer.create_host_features(df)
        
        expected = 1 if country == 'FRA' else 0
        assert result.iloc[0]['is_host'] == expected
