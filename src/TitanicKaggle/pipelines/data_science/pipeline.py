from kedro.pipeline import node, Pipeline
from TitanicKaggle.pipelines.data_science.nodes import (
    build_feature_table,
    create_label_table,
    create_model_table,
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
        ]
    )