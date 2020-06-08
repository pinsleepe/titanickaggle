from kedro.pipeline import node, Pipeline
from TitanicKaggle.pipelines.data_science.nodes import (
    build_feature_table,
    create_label_table,
    create_model_table,
    split_data,
    parameter_tuning_randomized_search,
    parameter_tuning_grid,
    model_fit_and_performance
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=build_feature_table,
                inputs="primary_table",
                outputs="feature_table",
                name="creating_feature_table",
            ),
            node(
                func=create_label_table,
                inputs="feature_table",
                outputs="label_table",
                name="creating_label_table",
            ),
            node(
                func=create_model_table,
                inputs="label_table",
                outputs="model_table",
                name="creating_model_table",
            ),
            node(
                func=split_data,
                inputs=["model_table", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="splitting_datae",
            ),
            node(
                func=parameter_tuning_randomized_search,
                inputs=["X_train", "y_train"],
                outputs="best_rfc_rs",
                name="tunning_RFC_RandomizedSearchCV",
            ),
            node(
                func=parameter_tuning_grid,
                inputs=["X_train", "y_train"],
                outputs="best_rfc_g",
                name="tunning_RFC_GridSearchCV",
            ),
            node(
                func=model_fit_and_performance,
                inputs=["best_rfc_rs", "X_train", "y_train", "X_test", "y_test"],
                outputs="predictions",
                name="model_performance",
            ),
        ]
    )