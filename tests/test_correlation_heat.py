import pytest
import sys
import os
import pandas as pd
import altair as alt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.correlation_heat import correlation_heat

sample_correlation_long = pd.DataFrame({
    "feature_1": ["A", "B", "B", "A"],
    "feature_2": ["A", "A", "B", "B"],
    "correlation": [1, 0.7, 1, 0.7],
})


#test 1: to test the correlation_heat function with a valid dataframe
#the function should return an Altair Chart object
def test_correlation_heat_success():
    feature_1 = 'feature_1'
    feature_2 = 'feature_2'
    correlation = 'correlation'

    chart = correlation_heat(sample_correlation_long, feature_1, feature_2, correlation)
    assert isinstance(chart, alt.Chart) 


#test 2: test the function with an invalid dataframe column name input
def test_correlation_heat_invalid_column_name():

    with pytest.raises(KeyError):
        correlation_heat(sample_correlation_long, 'invalid1', 'invalid2', 'invalid3')

#test 3: test the function with an empty dateframe input
def test_correlation_heat_empty_df():
    empty_df = pd.DataFrame()
    feature_1 = 'feature_1'
    feature_2 = 'feature_2'
    correlation = 'correlation'

    with pytest.raises(ValueError, match = 'DataFrame must contain observations'):
        correlation_heat(empty_df, feature_1, feature_2, correlation)


#test 4: test the function with an empty feature names
def test_correlation_heat_feature_names():
    feature_1 = ''
    feature_2 = 'feature_2'
    correlation = 'correlation'

    with pytest.raises(ValueError, match = "Feature names and correlation column must be non-empty strings."):
        correlation_heat(sample_correlation_long, feature_1, feature_2, correlation)


#test 5: test the function with an empty correlation column name
def test_correlation_heat_column_names():
    feature_1 = 'feature_1'
    feature_2 = 'feature_2'
    correlation = ''

    with pytest.raises(ValueError, match = "Feature names and correlation column must be non-empty strings."):
        correlation_heat(sample_correlation_long, feature_1, feature_2, correlation)


















    
