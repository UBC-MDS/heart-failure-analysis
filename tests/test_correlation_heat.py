import pytest
import sys
import os
import pandas as pd
import altair as alt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.correlation_heat import correlation_heat

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "feature_1": ["A", "B", "B", "A"],
        "feature_2": ["A", "A", "B", "B"],
        "correlation": [1, 0.7, 1, 0.7],
    })


#test 1: to test the correlation_heat function with a valid dataframe
def test_correlation_heat_success(sample_correlation_long):
    feature_1 = 'feature_1',
    feature_2 = 'feature_2',
    correlation = 'correlation'

    chart = correlation_heat(sample_correlation_long, feature_1, feature_2, correlation)
    assert isinstance(chart, alt.Chart) #the function should return an Altair Chart object


#test 2: test the function with an invalid dataframe input
def test_correlation_heat_invalid_df():
    invalid_df = {'feature_1': [1, 2, 3]}
    feature_1 = 'feature_1',
    feature_2 = 'feature_2',
    correlation = 'correlation'

    with pytest.raise(TypeError, match = 'Input must be a pandas DataFrame'):
        correlation_heat(invalid_input, feature_1, feature_2, correlation)

#test 3: test the function with an empty dateframe input
def test_correlation_heat_empty_df():
    empty_df = pd.DaraFrame()
    feature_1 = 'feature_1',
    feature_2 = 'feature_2',
    correlation = 'correlation'

    with pytest.raise(TypeError, match = 'DataFrame must contain observations'):
        correlation_heat(empty_df, feature_1, feature_2, correlation)


#test 4: test the function with an empty feature names
def test_correlation_heat_empty_feature_names(sample_correlation_long):
    empty_df = pd.DaraFrame()
    feature_1 = '',
    feature_2 = 'feature_2',
    correlation = 'correlation'

    with pytest.raise(TypeError, match = 'Feature columns and correlation column names must not by empty strings.'):
        correlation_heat(sample_correlation_long, feature_1, feature_2, correlation)


#test 5: test the function with an empty correlation column name
def test_correlation_heat_empty_feature_names(sample_correlation_long):
    empty_df = pd.DaraFrame()
    feature_1 = 'feature_1',
    feature_2 = 'feature_2',
    correlation = ''

    with pytest.raise(TypeError, match = 'Feature columns and correlation column names must not by empty strings.'):
        correlation_heat(sample_correlation_long, feature_1, feature_2, correlation)


















    
