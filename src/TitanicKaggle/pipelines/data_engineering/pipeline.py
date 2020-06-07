from kedro.pipeline import node, Pipeline
from TitanicKaggle.pipelines.data_engineering.nodes import (
    create_facility,
    preprocess_patients,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=create_facility,
                inputs="patients",
                outputs="facilities",
                name="creating_facilities",
            ),
            node(
                func=preprocess_patients,
                inputs=['patients', 'facilities'],
                outputs="preprocessed_patients",
                name="preprocessing_patients",
            ),
        ]
    )