import os
import sys
import pytest
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.download_and_convert import convert_binary_columns

@pytest.fixture
def tmp_csv_file(tmp_path):
    # Create a sample CSV file in a temporary directory
    data = pd.DataFrame({
        "binary_col": [0, 1, 1, 0],     # two unique values, should become bool
        "single_value_col": [1, 1, 1, 1], # one unique value, should remain unchanged
        "multi_val_col": [10, 20, 30, 40] # multiple unique values, remain unchanged
    })
    csv_file = tmp_path / "test.csv"
    data.to_csv(csv_file, index=False)
    return csv_file

def test_convert_binary_columns(tmp_csv_file, tmp_path):
    # Convert binary columns
    output_file = convert_binary_columns(str(tmp_csv_file), str(tmp_path))
    converted = pd.read_csv(output_file)

    # Check data types:
    # binary_col should be bool
    assert converted["binary_col"].dtype == bool

    # single_value_col should remain as numeric/int since it's not truly binary (only one unique value)
    # The code checks for exactly two unique values, so single_value_col will NOT be converted
    assert converted["single_value_col"].dtype != bool

    # multi_val_col has multiple unique values and should remain unchanged
    assert converted["multi_val_col"].dtype != bool
