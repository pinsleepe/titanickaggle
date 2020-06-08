from kedro.pipeline import node, Pipeline
from TitanicKaggle.pipelines.data_science.nodes import (
    build_feature_table,
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
        ]
    )