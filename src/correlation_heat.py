import os
import pandas as pd
import altair as alt

def correlation_heat(correlation_long: pd.DataFrame, feature_1: str, feature_2: str, correlation: str):
    """
    Create a correlation heatmap from a Pandas DataFrame.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        A long-format dataFrame containing correlation data.
    feature_1: a string
        The column name representing the first feature.
    feature_2: a string
        The column name representing the second feature.
    correlation: a string
        The column representing the correlation values. 

    Returns
    -------
    altair object
        The generated Altair heatmap chart. 

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    ValueError
        If the dataframe is empty.
    ValueError
        If the feature_1, feature_2, or correlation are empty strings. 
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")    
    if dataframe.empty:
        raise ValueError("DataFrame must contain observations.")
    if not all([feature_1, feature_2, correlation]):
        raise ValueError("Feature names and correlation column must be non-empty strings.")
    
    chart = alt.Chart(correlation_long).mark_rect().encode(
        x=f'{feature_1}:O',
        y=f'{feature_2}:O',
        color=alt.Color(f'{correlation}:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=[feature_1, feature_2, correlation]
    ).properties(
        width=600,
        height=600,
        title="Correlation Heatmap"
    )

    return chart
