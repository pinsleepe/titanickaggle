from kedro.pipeline import node, Pipeline
from TitanicKaggle.pipelines.data_engineering.nodes import (
    create_facility,
    preprocess_patients,
    remove_unnamed_column,
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
            node(
                func=remove_unnamed_column,
                inputs="immunization",
                outputs="preprocessed_immunization",
                name="creating_immunization",
            ),
        ]
    )