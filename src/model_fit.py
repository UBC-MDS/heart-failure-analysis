from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.pipeline import make_pipeline

def model_fit(model, preprocessor, grid, heart_failure_train):
    """
    Create a pipeline, tune hyperparameters using GridSearchCV with 10-fold cross-validation,
    and return the best-fitted model.

    Parameters
    ----------
    model : sklearn model instance
        The machine learning model to be tuned and fitted (e.g., LogisticRegression, KNeighborsClassifier).
    preprocessor : sklearn ColumnTransformer
        The preprocessing pipeline to be applied to the data.
    grid : dict
        Hyperparameter grid for tuning (e.g., {"logisticregression__C": [0.1, 1, 10]}).
    heart_failure_train : pandas DataFrame
        The training dataset including the target column 'DEATH_EVENT'.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A fitted pipeline with the best hyperparameters and the input model.
    """
    # Input Checks
    if model is None or not hasattr(model, "fit"):
        raise ValueError("A valid sklearn model with a 'fit' method must be provided.")

    if heart_failure_train.empty:
        raise ValueError("Input data cannot be empty.")

    if 'DEATH_EVENT' not in heart_failure_train.columns:
        raise KeyError("'DEATH_EVENT' column is missing in the input data.")


    pipeline = make_pipeline(
        preprocessor, 
        model
    )

    grid_search = GridSearchCV(
         pipeline,
         grid,
         cv=10,
         n_jobs=-1,
         return_train_score=True
         )
    grid_search.fit(
         heart_failure_train.drop(columns=['DEATH_EVENT']), 
         heart_failure_train['DEATH_EVENT']
         )
     
    return grid_search.best_estimator_, grid_search