from fastapi import HTTPException
from typing import List

from schemas import SinglePredictionResult


from .prediction_handlers import (
    regression_graph_predict,
    binary_graph_predict,
    multiclass_graph_predict,

    regression_graph_with_external_predict,
    binary_graph_with_external_predict,
    multiclass_graph_with_external_predict,

    regression_fc_predict,
    binary_fc_predict,
    multiclass_fc_predict,
)


def handle_prediction(model_data: dict, input_values: list) -> List[SinglePredictionResult]:
    """
    Handles predictions based on the model type provided in `model_data`.

    Args:
    - model_data (dict): A dictionary containing model information.
    - input_values (list): A list of inputs to use for prediction.

    Returns:
    - List[SinglePredictionResult]: A list of prediction results.

    Raises:
    - HTTPException(404): If the model type is unknown or unsupported.

    Notes on model_data['additional_model_params']:
        For each model type, the following additional parameters are expected in the database:
        
        TORCH-regression-graph-model:
            - featurizer (str): The featurizer.
            - normalization_mean (float): The target's normalization mean.
            - normalization_std (float): The target's normalization standard deviation.
        TORCH-binary-graph-model:
            - featurizer (str): The featurizer.
            - decision_threshold (float): The decision threshold.
        TORCH-multiclass-graph-model:
            - featurizer (str): The featurizer.
        TORCH-regression-graph-model-with-external:
            - featurizer (str): The featurizer.
            - external_preprocessor (str): The external preprocessor.
            - normalization_mean (float): The target's normalization mean.
            - normalization_std (float): The target's normalization standard deviation.
        TORCH-binary-graph-model-with-external:
            - featurizer (str): The featurizer.
            - external_preprocessor (str): The external preprocessor.
            - decision_threshold (float): The decision threshold.
        TORCH-multiclass-graph-model-with-external:
            - featurizer (str): The featurizer.
            - external_preprocessor (str): The external preprocessor.
        TORCH-binary-fc-model:
            - preprocessor (str): The preprocessor.
            - decision_threshold (float): The decision threshold.
        TORCH-regression-fc-model:
            - preprocessor (str): The preprocessor.
            - normalization_mean (float): The target's normalization mean.
            - normalization_std (float): The target's normalization standard deviation.
        TORCH-multiclass-fc-model:
            - preprocessor (str): The preprocessor.
    """

    model_type = model_data['type']

    prefix = "TORCH-"
    if model_type.startswith(prefix):
        model_type[len(prefix):]
    
    model_type = model_type.removeprefix("TORCH-")
    
    match model_type:
        case 'regression-graph-model':
            return regression_graph_predict(model_data, input_values)
        case 'binary-graph-model':
            return binary_graph_predict(model_data, input_values)
        case "multiclass-graph-model":
            return multiclass_graph_predict(model_data, input_values)
        case 'regression-graph-model-with-external':
            return regression_graph_with_external_predict(model_data, input_values)
        case 'binary-graph-model-with-external':
            return binary_graph_with_external_predict(model_data, input_values)
        case "multiclass-graph-model-with-external":
            return multiclass_graph_with_external_predict(model_data, input_values)
        case 'binary-fc-model':
            return binary_fc_predict(model_data, input_values)
        case 'regression-fc-model':
            return regression_fc_predict(model_data, input_values)
        case 'multiclass-fc-model':
            return multiclass_fc_predict(model_data, input_values)
        case _:
            raise HTTPException(status_code=404, detail=f"Can't make prediction of model with model_type {model_type}.")