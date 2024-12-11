import pytest
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.split_data import split_data
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.split_data import split_data

# Sample valid data
valid_data = pd.DataFrame({
    "age": [50, 60, 70, 80],
    "anaemia": [1, 0, 0, 1],
    "creatinine_phosphokinase": [120, 450, 232, 1500],
    "diabetes": [0, 1, 0, 0],
    "ejection_fraction": [35, 50, 25, 40],
    "high_blood_pressure": [1, 0, 1, 0],
    "platelets": [250000, 300000, 400000, 200000],
    "serum_creatinine": [1.5, 1.8, 1.2, 2.5],
    "serum_sodium": [135, 140, 145, 138],
    "sex": [1, 0, 1, 1],
    "smoking": [0, 1, 0, 0],
    "time": [30, 45, 100, 210],
    "DEATH_EVENT": [1, 0, 0, 1]
})

# Fixture to create a temporary directory for outputs
@pytest.fixture
def temp_dir(tmpdir):
    return tmpdir

# Test: Valid split
def test_split_data_valid(temp_dir):
    split_data(valid_data, temp_dir, train_size=0.75, random_state=42)

    # Check that output files are created
    train_file = os.path.join(temp_dir, "heart_failure_train.csv")
    test_file = os.path.join(temp_dir, "heart_failure_test.csv")
    assert os.path.exists(train_file), "Training file not created."
    assert os.path.exists(test_file), "Testing file not created."

    # Check train/test split sizes
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    assert len(train_data) == 3, "Incorrect training data size."
    assert len(test_data) == 1, "Incorrect testing data size."

    # Verify stratification
    assert train_data['DEATH_EVENT'].mean() == pytest.approx(
        valid_data['DEATH_EVENT'].mean(), abs=0.1
    ), "Stratification not preserved."

# Test: Invalid train_size
def test_split_data_invalid_train_size(temp_dir):
    with pytest.raises(ValueError, match="train_size must be between 0 and 1."):
        split_data(valid_data, temp_dir, train_size=1.5)

# Test: Missing DEATH_EVENT column
def test_split_data_missing_column(temp_dir):
    data_missing_column = valid_data.drop(columns=["DEATH_EVENT"])
    with pytest.raises(ValueError, match="The input data must contain the 'DEATH_EVENT' column"):
        split_data(data_missing_column, temp_dir)

# Test: Empty DataFrame
def test_split_data_empty_dataframe(temp_dir):
    empty_data = valid_data.iloc[0:0]
    with pytest.raises(ValueError, match="Found array with 0 sample"):
        split_data(empty_data, temp_dir)

# Test: Random state consistency
def test_split_data_random_state(temp_dir):
    split_data(valid_data, temp_dir, random_state=42)
    train_file_1 = os.path.join(temp_dir, "heart_failure_train.csv")
    train_data_1 = pd.read_csv(train_file_1)

    split_data(valid_data, temp_dir, random_state=42)
    train_file_2 = os.path.join(temp_dir, "heart_failure_train.csv")
    train_data_2 = pd.read_csv(train_file_2)

    # Verify the splits are identical
    pd.testing.assert_frame_equal(train_data_1, train_data_2, check_like=True)

# Test: Directory creation
def test_split_data_directory_creation(temp_dir):
    output_dir = os.path.join(temp_dir, "new_subdir")
    split_data(valid_data, output_dir)
    assert os.path.exists(output_dir), "Output directory not created."
    assert os.path.exists(os.path.join(output_dir, "heart_failure_train.csv")), "Training file not created in directory."
    assert os.path.exists(os.path.join(output_dir, "heart_failure_test.csv")), "Test file not created in directory."

# Test: Imbalanced classes
def test_split_data_imbalanced_classes(temp_dir):
    imbalanced_data = valid_data.copy()
    imbalanced_data["DEATH_EVENT"] = [0, 0, 0, 1]
    split_data(imbalanced_data, temp_dir, random_state=42)

    train_file = os.path.join(temp_dir, "heart_failure_train.csv")
    train_data = pd.read_csv(train_file)

    # Verify stratification works with imbalanced data
    assert train_data['DEATH_EVENT'].mean() == pytest.approx(
        imbalanced_data['DEATH_EVENT'].mean(), abs=0.1
    ), "Stratification failed with imbalanced data."
