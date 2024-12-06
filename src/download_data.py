# download_data_chunk_1.py
# author: Merari Santana-Carbajal
# date: 2024-12-03

import click
import os
import zipfile
import requests


def read_zip(url, directory):
    """
    Downloads a ZIP file from the given URL, saves it to the specified directory, and extracts its contents.

    Parameters:
    ----------
    url : str
        The URL of the ZIP file to download.
    directory : str
        The directory where the ZIP file will be saved and extracted.

    Returns:
    -------
    None

    Raises:
    ------
    ValueError:
        If the URL does not exist, is not a ZIP file, or the directory does not exist.
    """
    # Validate URL and fetch the file
    request = requests.get(url)
    filename_from_url = os.path.basename(url)

    if request.status_code != 200:
        raise ValueError(f"The URL '{url}' does not exist or is inaccessible.")
    
    if not url.endswith('.zip'):
        raise ValueError(f"The URL '{url}' does not point to a ZIP file.")
    
    if not os.path.isdir(directory):
        raise ValueError(f"The directory '{directory}' does not exist.")
    
    # Save the ZIP file
    path_to_zip_file = os.path.join(directory, filename_from_url)
    with open(path_to_zip_file, 'wb') as f:
        f.write(request.content)

    # Extract the ZIP file
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory)
    
    # Validate extraction
    extracted_files = os.listdir(directory)
    if len(extracted_files) == 0:
        raise ValueError("The ZIP file appears to be empty or extraction failed.")
    print(f"Successfully extracted {len(extracted_files)} files to '{directory}'.")


@click.command()
@click.option('--url', type=str, required=True, help="URL of the ZIP dataset to be downloaded.")
@click.option('--write_to', type=str, required=True, help="Path to the directory where data will be saved.")
def main(url, write_to):
    """
    Downloads and extracts a ZIP file from a given URL to a specified directory.

    If the directory does not exist, it will be created automatically.
    """
    # Ensure the directory exists
    if not os.path.exists(write_to):
        os.makedirs(write_to, exist_ok=True)
    
    try:
        read_zip(url, write_to)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == '__main__':
    main()
