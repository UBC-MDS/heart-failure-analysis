# modelling.py
# author: Gurmehak Kaur 
# date: 2024-12-06

import click
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import altair as alt

@click.command()
@click.option('--training-data', type=str, help="Path to training data")
@click.option('--pipeline-to', type=str, help="Path to directory where the final pipeline object will be written to")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=522)
def main(training_data, pipeline_to, plot_to, seed):
    '''Tests three pipelines for heart failure prediction and selects Logistic Regression as the final model.'''
    
    os.makedirs(pipeline_to, exist_ok=True)
    os.makedirs(plot_to, exist_ok=True)
    
    np.random.seed(seed)

    # Load the dataset
    heart_failure_train = pd.read_csv(training_data)
    
    numeric_columns = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']
    binary_columns = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
    
    heart_failure_preprocessor = make_column_transformer(
        (StandardScaler(), numeric_columns),
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop='if_binary', dtype=int), binary_columns),
        remainder='passthrough'
    )


    # ----- Decision Tree Pipeline -----
    dt_pipeline = make_pipeline(
        heart_failure_preprocessor, 
        DecisionTreeClassifier(random_state=seed)
    )
    dt_scores = cross_validate(
        dt_pipeline, 
        heart_failure_train.drop(columns=['DEATH_EVENT']), 
        heart_failure_train['DEATH_EVENT'], 
        return_train_score=True
    )
    dt_scores = pd.DataFrame(dt_scores).sort_values('test_score', ascending=False)


    # ----- K-Nearest Neighbors Pipeline -----
    knn_param_grid = {"kneighborsclassifier__n_neighbors": range(1, 100, 3)}

    # Hyperparameter tuning and model fitting 
    knn_best_model, knn_cv_results = model_fit(
        KNeighborsClassifier(),
        heart_failure_preprocessor,
        knn_param_grid,
        heart_failure_train
    )


    # ----- Logistic Regression Pipeline -----
    lr_param_grid = {"logisticregression__C": 10.0 ** np.arange(-5, 5, 1)}

    # Hyperparameter tuning and model fitting 
    lr_best_model, lr_cv_results = model_fit(
        LogisticRegression(random_state=123, max_iter=2000, class_weight="balanced"),
        heart_failure_preprocessor,
        lr_param_grid,
        heart_failure_train
    )

    print("Best Logistic Regression Model:", lr_best_model)


    # Save the Logistic Regression pipeline
    with open(os.path.join(pipeline_to, "heart_failure_model.pickle"), 'wb') as f:
        pickle.dump(heart_failure_model, f)

    # ----- Visualizing Logistic Regression Scores -----
    lr_scores = pd.DataFrame(lr_grid_search.cv_results_).sort_values(
        'mean_test_score', ascending=False
    )[['param_logisticregression__C', 'mean_test_score', 'mean_train_score']]

    lr_plot = alt.Chart(lr_scores).transform_fold(
        ["mean_test_score", "mean_train_score"],
        as_=["Score Type", "Score"]
    ).mark_line().encode(
        x=alt.X("param_logisticregression__C:Q", title="C (Regularization Parameter)", scale=alt.Scale(type='log')),
        y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0.75, 0.85])),
        color=alt.Color("Score Type:N", title="Score Type",
                        scale=alt.Scale(domain=["mean_test_score", "mean_train_score"], range=["skyblue", "pink"])),
        tooltip=["param_logisticregression__C", "Score Type:N", "Score:Q"]
    ).properties(
        title="Training vs Cross-Validation Scores (Log Scale)",
        width=600,
        height=400
    )
    lr_plot.save(os.path.join(plot_to, "logistic_regression_scores.html"), scale_factor=2.0)

    # ----- Analyzing Logistic Regression Coefficients -----
    lr_model = lr_best_model.named_steps['logisticregression']
    features = lr_model.coef_
    feature_names = heart_failure_train.drop(columns=['DEATH_EVENT']).columns
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': features[0],
        'Absolute_Coefficient': abs(features[0])
    }).sort_values(by='Absolute_Coefficient', ascending=False)
    coefficients.to_csv(os.path.join(plot_to, "logistic_regression_coefficients.csv"), index=False)
    print("Logistic Regression Coefficients:", coefficients)

if __name__ == '__main__':
    main()
