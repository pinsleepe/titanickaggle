from typing import Dict

from kedro.pipeline import Pipeline

from TitanicKaggle.pipelines.data_engineering import pipeline as de


def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    de_pipeline = de.create_pipeline()

    return {
        "de": de_pipeline,
        "__default__": de_pipeline,
    }