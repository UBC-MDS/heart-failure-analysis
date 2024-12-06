# data_exploration.py
# author: Merari Santana-Carbajal
# date: 2024-12-06

import os
import pandas as pd
import click
import altair_ally as aly

def explore_data(file_path):
    """
    Perform data exploration on the given dataset, including:
    - Shape and information
    - Value counts for target column
    - Summary statistics
    - Missing values check
    - Visualizations (heatmap, distributions, pair plot, correlation matrix, and parallel coordinates)

    Parameters:
    ----------
    file_path : str
        The file path of the dataset to be explored.

    Returns:
    -------
    None
    """
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise ValueError(f"The file {file_path} does not exist.")

    # Load the dataset
    heart_failure_data = pd.read_csv(file_path)

    # Display dataset shape
    print(f"Dataset Shape: {heart_failure_data.shape}")

    # Display dataset info
    print("\nDataset Info:")
    heart_failure_data.info()

    # Display value counts for the target column
    print("\nTarget Column Value Counts:")
    print(heart_failure_data['DEATH_EVENT'].value_counts())

    # Display summary statistics
    print("\nSummary Statistics:")
    print(heart_failure_data.describe())

    # Check for missing values
    missing_values = heart_failure_data.isnull().sum()
    print("\nMissing Values:")
    print(missing_values)

    # Heatmap visualization
    print("\nGenerating heatmap...")
    aly.heatmap(heart_failure_data, color="DEATH_EVENT")

    # Distribution plots for all columns
    print("\nVisualizing distributions for all columns...")
    aly.dist(heart_failure_data)

    # Pair plot
    print("\nGenerating pair plot...")
    aly.pair(heart_failure_data, color="DEATH_EVENT")

    # Correlation matrix
    print("\nGenerating correlation matrix...")
    aly.corr(heart_failure_data)

    # Parallel coordinates plot
    print("\nGenerating parallel coordinates plot...")
    aly.parcoord(heart_failure_data, color="DEATH_EVENT")

    # Distribution plot with color
    print("\nGenerating distribution plot with color...")
    aly.dist(heart_failure_data, color="DEATH_EVENT")

@click.command()
@click.option('--file_path', type=click.Path(exists=True, dir_okay=False, readable=True), help="Path to the dataset CSV file")
def main(file_path):
    """
    Performs exploratory data analysis on a dataset.
    """
    try:
        explore_data(file_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

if __name__ == '__main__':
    main()
