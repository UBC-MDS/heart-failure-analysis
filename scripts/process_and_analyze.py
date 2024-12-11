# process_and_analyze.py
# author: Merari Santana-Carbajal
# date: 2024-12-06

import os
import pandas as pd
import click
import altair_ally as aly
from sklearn.model_selection import train_test_split
import pandera as pa
from pandera import Check, Column, DataFrameSchema
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.split_data import split_data

def validate_data(file_path):
    """
    Validate the dataset against a predefined schema to ensure data integrity.
    """
    # Define the schema
    schema = pa.DataFrameSchema(
        {
            "age": Column(float, Check.between(1, 120), nullable=True),
            "anaemia": Column(bool),
            "creatinine_phosphokinase": Column(int, Check.between(20, 50000), nullable=True),
            "diabetes": Column(bool),
            "ejection_fraction": Column(int, Check.between(5, 90), nullable=True),
            "high_blood_pressure": Column(bool),
            "platelets": Column(float, Check.between(10000, 900000), nullable=True),
            "serum_creatinine": Column(float, Check.between(0.2, 10), nullable=True),
            "serum_sodium": Column(int, Check.between(110, 190), nullable=True),
            "sex": Column(bool),
            "smoking": Column(bool),
            "time": Column(int, Check.between(1, 360), nullable=True),
            "DEATH_EVENT": Column(bool)
        },
        checks=[
            Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found."),
            Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found.")
        ]
    )

    # Load the dataset
    heart_failure_data = pd.read_csv(file_path)

    # Validate the dataset
    schema.validate(heart_failure_data, lazy=True)
    print("Dataset validation successful!")

    return heart_failure_data


def explore_data(data):
    """
    Perform exploratory data analysis on the dataset.
    """
    print(f"\nDataset Shape: {data.shape}")
    print("\nDataset Info:")
    data.info()

    print("\nTarget Column Value Counts:")
    print(data['DEATH_EVENT'].value_counts())

    print("\nSummary Statistics:")
    print(data.describe())

    print("\nMissing Values:")
    print(data.isnull().sum())

    # Generate visualizations
    print("\nGenerating visualizations...")
    aly.heatmap(data, color="DEATH_EVENT")
    aly.dist(data)
    aly.pair(data, color="DEATH_EVENT")
    aly.corr(data)
    aly.parcoord(data, color="DEATH_EVENT")
    aly.dist(data, color="DEATH_EVENT")

split_data(data, output_dir, train_size = 0.8, random_stat = 522)


@click.command()
@click.option('--file_path', type=click.Path(exists=True, dir_okay=False), help="Path to the dataset CSV file.")
@click.option('--output_dir', default="../data/processed", help="Directory to save the split datasets.")
def main(file_path, output_dir):
    """
    Validate, analyze, and split a dataset.
    """
    try:
        # Step 1: Validate the dataset
        print("Validating dataset...")
        data = validate_data(file_path)

        # Step 2: Perform EDA
        print("\nPerforming exploratory data analysis...")
        explore_data(data)

        # Step 3: Split the dataset
        print("\nSplitting the dataset...")
        split_data(data, output_dir)

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


if __name__ == '__main__':
    main()
