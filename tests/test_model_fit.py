import pytest
import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_fit import model_fit

# create sample or mock data
@pytest.fixture
def mock_data():
    """Create a mock dataset for testing."""
#     np.random.seed(123)
    mock_data = pd.DataFrame({
        "age": np.random.randint(40, 80, 100),
        "creatinine_phosphokinase": np.random.randint(50, 300, 100),
        "ejection_fraction": np.random.randint(10, 60, 100),
        "platelets": np.random.uniform(100000, 400000, 100),
        "serum_creatinine": np.random.uniform(0.5, 2.5, 100),
        "serum_sodium": np.random.randint(120, 140, 100),
        "time": np.random.randint(0, 300, 100),
        "anaemia": np.random.choice([0, 1], 100),
        "diabetes": np.random.choice([0, 1], 100),
        "high_blood_pressure": np.random.choice([0, 1], 100),
        "sex": np.random.choice([0, 1], 100),
        "smoking": np.random.choice([0, 1], 100),
        "DEATH_EVENT": np.random.choice([0, 1], 100),
    })
    return pd.DataFrame(mock_data)

@pytest.fixture
def preprocessor():
    """Set up the preprocessing pipeline."""
    numeric_columns = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium", "time"]
    binary_columns = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]

    return make_column_transformer(
        (StandardScaler(), numeric_columns),
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop='if_binary', dtype=int), binary_columns),
        remainder="passthrough"
    )

# ------- Test cases -------

# Test: to test if function is working as expected on logistic regression example
def test_model_fit_logistic_regression(mock_data, preprocessor):
    """Test model_fit with Logistic Regression."""
    param_grid = {"logisticregression__C": [0.1, 1, 10]}
    model = LogisticRegression(max_iter=1000, random_state=123)

    best_model, cv_results = model_fit(model, preprocessor, param_grid, mock_data)

    # Assertions
    assert best_model is not None
    assert "logisticregression" in best_model.named_steps
    assert isinstance(cv_results, dict)
    assert "mean_test_score" in cv_results

# Test: to test if function is working as expected on KNN example
def test_model_fit_knn(mock_data, preprocessor):
    """Test model_fit with K-Nearest Neighbors."""
    param_grid = {"kneighborsclassifier__n_neighbors": [3, 5, 7]}
    model = KNeighborsClassifier()

    best_model, cv_results = model_fit(model, preprocessor, param_grid, mock_data)

    # Assertions
    assert best_model is not None
    assert "kneighborsclassifier" in best_model.named_steps
    assert isinstance(cv_results, dict)
    assert "mean_test_score" in cv_results

# Test: to raise error if incorrect model is passed to the function
def test_model_fit_invalid_model(mock_data, preprocessor):
    """Test model_fit with an invalid model."""
    param_grid = {"invalidmodel__param": [1, 2, 3]}

    with pytest.raises(ValueError, match="A valid sklearn model with a 'fit' method must be provided."):
        _ = model_fit(None, preprocessor, param_grid, mock_data)

# Test: to raise value error if input data is empty
def test_model_fit_empty_data(preprocessor):
    """Test model_fit with empty data."""
    empty_data = pd.DataFrame()

    param_grid = {"logisticregression__C": [0.1, 1, 10]}
    model = LogisticRegression(max_iter=1000, random_state=123)

    with pytest.raises(ValueError, match="Input data cannot be empty."):
        _ = model_fit(model, preprocessor, param_grid, empty_data)

# Test: to raise value error if parameter grid for random search is not valid
def test_model_fit_invalid_grid(mock_data, preprocessor):
    """Test model_fit with an invalid parameter grid."""

    param_grid = {"invalidparameter__C": [0.1, 1, 10]}
    model = LogisticRegression(max_iter=1000, random_state=123)

    with pytest.raises(ValueError, match="Invalid parameter"):
        _ = model_fit(model, preprocessor, param_grid, mock_data)

# Test: to check if output (cross validation scores table) is producing correct columns 
def test_model_fit_cross_validation_results(mock_data, preprocessor):
    """Verify cross-validation results contain expected fields."""
    param_grid = {"logisticregression__C": [0.1, 1, 10]}
    model = LogisticRegression(max_iter=1000, random_state=123)

    _, cv_results = model_fit(model, preprocessor, param_grid, mock_data)

    # Assertions for CV results
    expected_keys = ["mean_test_score", "std_test_score", "mean_train_score", "std_train_score"]
    for key in expected_keys:
        assert key in cv_results