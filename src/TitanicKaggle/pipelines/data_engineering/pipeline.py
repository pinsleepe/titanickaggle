from kedro.pipeline import node, Pipeline
from TitanicKaggle.pipelines.data_engineering.nodes import (
    create_facility,
    preprocess_patients,
    preprocess_immunization,
    build_primary_table,
    remove_bad_pat_id
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
                func=preprocess_immunization,
                inputs=['immunization', 'preprocessed_patients'],
                outputs="preprocessed_immunization",
                name="preprocessing_immunization",
            ),
            node(
                func=remove_bad_pat_id,
                inputs=["preprocessed_patients", "preprocessed_immunization"],
                outputs="clean_patients",
                name="removing_patients_with_no_immunization",
            ),
            node(
                func=build_primary_table,
                inputs=["clean_patients", "preprocessed_immunization"],
                outputs="primary_table",
                name="creating_primary_table",
            ),
        ]
    )