import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data, output_dir, train_size=0.8, random_state=522):
    """
    Split the dataset into training and testing sets and save them as CSV files.

    This function splits the input dataset into training and testing subsets, 
    ensuring the `DEATH_EVENT` column is stratified to maintain class distribution 
    in both sets. The resulting subsets are saved as CSV files in the specified directory.

    Parameters:
    ----------
    data : pd.DataFrame
        The input dataset as a pandas DataFrame. It must contain the `DEATH_EVENT` column 
        for stratified splitting.

    output_dir : str
        The directory where the split CSV files will be saved. If the directory does 
        not exist, it will be created.

    train_size : float, optional (default=0.8)
        The proportion of the data to include in the training set. The remaining 
        proportion will be included in the test set.

    random_state : int, optional (default=522)
        The random seed for reproducibility of the split.

    Returns:
    -------
    None

    Saves:
    ------
    - `heart_failure_train.csv`: The training dataset.
    - `heart_failure_test.csv`: The testing dataset.

    Raises:
    ------
    ValueError:
        If `train_size` is not between 0 and 1 or if `DEATH_EVENT` is missing in the dataset.

    Examples:
    --------
    >>> df = pd.read_csv("heart_failure_clinical_records.csv")
    >>> split_data(df, "output", train_size=0.75, random_state=42)
    Training data saved to: output/heart_failure_train.csv
    Test data saved to: output/heart_failure_test.csv
    """
    # Validate train_size
    if not (0 < train_size < 1):
        raise ValueError("train_size must be between 0 and 1.")

    # Ensure DEATH_EVENT column exists
    if 'DEATH_EVENT' not in data.columns:
        raise ValueError("The input data must contain the 'DEATH_EVENT' column for stratified splitting.")

    # Split the data
    train_data, test_data = train_test_split(
        data,
        train_size=train_size,
        stratify=data['DEATH_EVENT'],
        random_state=random_state
    )

    # Save the splits
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "heart_failure_train.csv")
    test_path = os.path.join(output_dir, "heart_failure_test.csv")

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"Training data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")
