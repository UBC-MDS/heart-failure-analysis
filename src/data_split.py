# data_split.py
# author: Merari Santana-Carbajal
# date: 2024-12-06

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import click

def split_data(file_path, output_dir, train_size=0.8, random_state=522):
    """
    Split the dataset into training and test sets, stratified by the target column.

    Parameters:
    ----------
    file_path : str
        Path to the input dataset CSV file.
    output_dir : str
        Directory to save the split datasets.
    train_size : float, optional (default=0.8)
        Proportion of the dataset to include in the training set.
    random_state : int, optional (default=522)
        Random seed for reproducibility.

    Returns:
    -------
    None
    """
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise ValueError(f"The file {file_path} does not exist.")

    # Load the dataset
    heart_failure_data = pd.read_csv(file_path)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Split the data into training and test sets
    heart_failure_train, heart_failure_test = train_test_split(
        heart_failure_data, 
        train_size=train_size, 
        stratify=heart_failure_data['DEATH_EVENT'], 
        random_state=random_state
    )

    # Save the split datasets
    train_file = os.path.join(output_dir, 'heart_failure_train.csv')
    test_file = os.path.join(output_dir, 'heart_failure_test.csv')

    heart_failure_train.to_csv(train_file, index=False)
    heart_failure_test.to_csv(test_file, index=False)

    print(f"Training data saved to: {train_file}")
    print(f"Test data saved to: {test_file}")

@click.command()
@click.option('--file_path', type=click.Path(exists=True, dir_okay=False, readable=True), help="Path to the dataset CSV file")
@click.option('--output_dir', type=click.Path(file_okay=False, writable=True), help="Directory to save the split datasets")
@click.option('--train_size', type=float, default=0.8, help="Proportion of the dataset to include in the training set (default=0.8)")
@click.option('--random_state', type=int, default=522, help="Random seed for reproducibility (default=522)")
def main(file_path, output_dir, train_size, random_state):
    """
    Splits a dataset into training and test sets.
    """
    try:
        split_data(file_path, output_dir, train_size, random_state)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

if __name__ == '__main__':
    main()
