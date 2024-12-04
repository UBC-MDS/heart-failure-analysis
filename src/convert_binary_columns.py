# convert_binary_columns.py
# author: Merari Santana-Carbajal
# date: 2023-11-27

import os
import pandas as pd
import click

def convert_binary_columns(file_path, binary_columns):
    """
    Convert specified binary columns in a dataset to boolean values (True/False).

    Parameters:
    ----------
    file_path : str
        The file path of the dataset to be processed.
    binary_columns : list
        List of column names that should be converted to boolean.

    Returns:
    -------
    None
    """
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise ValueError(f"The file {file_path} does not exist.")

    # Load the dataset
    heart_failure_data = pd.read_csv(file_path)

    # Check if all binary columns exist in the dataset
    missing_columns = [col for col in binary_columns if col not in heart_failure_data.columns]
    if missing_columns:
        raise ValueError(f"The following columns are missing from the dataset: {missing_columns}")

    # Convert binary columns to True/False
    heart_failure_data[binary_columns] = heart_failure_data[binary_columns].astype(bool)

    # Save the modified dataset back to a CSV file
    output_file = os.path.splitext(file_path)[0] + "_converted.csv"
    heart_failure_data.to_csv(output_file, index=False)

    print(f"Binary columns converted and saved to {output_file}")

@click.command()
@click.option('--file_path', type=str, help="Path to the dataset CSV file")
@click.option('--binary_columns', type=str, help="Comma-separated list of binary columns to convert")
def main(file_path, binary_columns):
    """
    Converts specified binary columns in a dataset to boolean values (True/False).
    """
    # Convert the binary_columns string into a list
    binary_columns_list = binary_columns.split(',')
    
    try:
        convert_binary_columns(file_path, binary_columns_list)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
