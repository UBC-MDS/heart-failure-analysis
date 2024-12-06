# fit_heart_failure_model.py
# author: Gurmehak
# date: 2024-12-03

import click
import os
import altair as alt
import numpy as np
import pandas as pd
import pickle
from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureFeatureCorrelation
from deepchecks.tabular import Dataset
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="deepchecks")

@click.command()
@click.option('--training-data', type=str, help="Path to training data")
@click.option('--preprocessor', type=str, help="Path to preprocessor object")
@click.option('--pipeline-to', type=str, help="Path to directory where the pipeline object will be written to")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=522)
def main(training_data, preprocessor, columns_to_drop, pipeline_to, plot_to, seed):
    '''Fits a heart failure prediction model to the training data 
    and saves the pipeline object.'''
    np.random.seed(seed)

    # # Load training data and preprocessor
    # heart_failure_train = pd.read_csv(training_data)
    # heart_failure_preprocessor = pickle.load(open(preprocessor, "rb"))

    # # Validate training data
    # scaled_train_ds = Dataset(heart_failure_train, label="DEATH_EVENT", cat_features=[])

    # check_feat_lab_corr = FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.9)
    # check_feat_lab_corr_result = check_feat_lab_corr.run(dataset=scaled_train_ds)

    # check_feat_feat_corr = FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(
    #     threshold=0.92, n_pairs=0)
    # check_feat_feat_corr_result = check_feat_feat_corr.run(dataset=scaled_train_ds)

    # if not check_feat_lab_corr_result.passed_conditions():
    #     raise ValueError("Feature-Label correlation exceeds the maximum acceptable threshold.")

    # if not check_feat_feat_corr_result.passed_conditions():
    #     raise ValueError("Feature-feature correlation exceeds the maximum acceptable threshold.")

    # Build pipeline and tune model
    pipeline = make_pipeline(
        heart_failure_preprocessor,
        LogisticRegression(random_state=seed, max_iter=2000, class_weight="balanced")
    )

    param_grid = {
        "logisticregression__C": 10.0 ** np.arange(-5, 5, 1)
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        n_jobs=-1,
        return_train_score=True
    )

    heart_failure_fit = grid_search.fit(
        heart_failure_train.drop(columns=["DEATH_EVENT"]),
        heart_failure_train["DEATH_EVENT"]
    )

    with open(os.path.join(pipeline_to, "heart_failure_pipeline.pickle"), 'wb') as f:
        pickle.dump(heart_failure_fit, f)

    # Extract and plot scores
    scores = pd.DataFrame(grid_search.cv_results_).sort_values(
        'mean_test_score', ascending=False)[['param_logisticregression__C', 'mean_test_score', 'mean_train_score']]

    plot = alt.Chart(scores).transform_fold(
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

    plot.save(os.path.join(plot_to, "heart_failure_scores.png"), scale_factor=2.0)

    # Extract model coefficients
    lr_best_model = grid_search.best_estimator_.named_steps['logisticregression']
    features = lr_best_model.coef_
    feature_names = heart_failure_train.drop(columns=['DEATH_EVENT']).columns
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': features[0],
        'Absolute_Coefficient': abs(features[0])
    }).sort_values(by='Absolute_Coefficient', ascending=False)

    coefficients.to_csv(os.path.join(plot_to, "heart_failure_coefficients.csv"), index=False)

if __name__ == '__main__':
    main()
