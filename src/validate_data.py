# validate_data.py
# author: Merari Santana-Carbajal
# date: 2024-12-06

import os
import pandas as pd
import click
import pandera as pa
from pandera import Check, Column, DataFrameSchema

def validate_data(file_path):
    """
    Validate the dataset against a predefined schema to ensure data integrity.

    Parameters:
    ----------
    file_path : str
        The file path of the dataset to validate.

    Returns:
    -------
    None

    Raises:
    ------
    ValueError:
        If the dataset fails the validation checks.
    """
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise ValueError(f"The file {file_path} does not exist.")

    # Load the dataset
    heart_failure_data = pd.read_csv(file_path)

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

    # Validate the dataset
    try:
        schema.validate(heart_failure_data, lazy=True)
        print("Dataset validation successful!")
    except pa.errors.SchemaErrors as e:
        print("Dataset validation failed!")
        print(e)

@click.command()
@click.option('--file_path', type=click.Path(exists=True, dir_okay=False, readable=True), help="Path to the dataset CSV file")
def main(file_path):
    """
    Validates a dataset against a predefined schema.
    """
    try:
        validate_data(file_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

if __name__ == '__main__':
    main()
