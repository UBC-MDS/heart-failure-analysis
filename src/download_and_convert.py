# download_and_convert.py
# author: Merari Santana-Carbajal
# date: 2024-12-06

import os
import pandas as pd
import requests
import zipfile
import click


def download_and_extract_zip(url, directory):
    """
    Download a ZIP file from the given URL and extract its contents to a specified directory.

    Parameters:
    ----------
    url : str
        The URL of the ZIP file to download.
    directory : str
        The directory where the ZIP file will be saved and extracted.

    Returns:
    -------
    str: Path to the extracted CSV file.
    """
    # Validate URL and fetch the file
    request = requests.get(url)
    filename_from_url = os.path.basename(url)

    if request.status_code != 200:
        raise ValueError(f"The URL '{url}' does not exist or is inaccessible.")
    
    if not url.endswith('.zip'):
        raise ValueError(f"The URL '{url}' does not point to a ZIP file.")
    
    # Ensure the output directory exists
    os.makedirs(directory, exist_ok=True)

    # Save the ZIP file
    path_to_zip_file = os.path.join(directory, filename_from_url)
    with open(path_to_zip_file, 'wb') as f:
        f.write(request.content)

    # Extract the ZIP file
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory)

    # Find the first CSV file in the extracted files
    extracted_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    if len(extracted_files) == 0:
        raise ValueError("The ZIP file appears to contain no CSV files.")
    
    print(f"Successfully extracted {len(extracted_files)} files to '{directory}'.")
    return os.path.join(directory, extracted_files[0])


def convert_binary_columns(file_path, output_dir):
    """
    Automatically detect and convert binary columns in a dataset to boolean values (True/False).

    Parameters:
    ----------
    file_path : str
        The file path of the dataset to process.
    output_dir : str
        The directory where the converted dataset will be saved.

    Returns:
    -------
    str: Path to the converted dataset.
    """
    # Load the dataset
    heart_failure_data = pd.read_csv(file_path)

    # Automatically detect binary columns
    binary_columns = [
        col for col in heart_failure_data.columns
        if heart_failure_data[col].dropna().nunique() == 2
    ]

    print(f"Detected binary columns: {binary_columns}")

    # Convert binary columns to True/False
    heart_failure_data[binary_columns] = heart_failure_data[binary_columns].astype(bool)

    # Save the converted dataset
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "heart_failure_clinical_records_dataset_converted.csv")
    heart_failure_data.to_csv(output_file, index=False)

    print(f"Binary columns converted and saved to {output_file}")
    return output_file


@click.command()
@click.option('--url', type=str, required=True, help="URL of the ZIP dataset to be downloaded.")
@click.option(
    '--write_to', 
    type=str, 
    default="../data", 
    help="Path to the directory where data will be saved (default is '../data')."
)
def main(url, write_to):
    """
    Download and extract a dataset from a ZIP file, then automatically detect and convert binary columns.
    """
    try:
        # Step 1: Download and extract the dataset
        extracted_csv = download_and_extract_zip(url, write_to)
        print(f"Dataset extracted to {extracted_csv}")

        # Step 2: Convert binary columns
        converted_file = convert_binary_columns(extracted_csv, write_to)
        print(f"Converted dataset saved at {converted_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


if __name__ == '__main__':
    main()
