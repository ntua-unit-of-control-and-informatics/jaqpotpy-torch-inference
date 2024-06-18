from fastapi import HTTPException

from .prediction_handlers import regression_graph_predict
from .prediction_handlers import binary_graph_predict
from .prediction_handlers import multiclass_graph_predict

from .prediction_handlers import regression_graph_with_external_predict
from .prediction_handlers import binary_graph_with_external_predict
from .prediction_handlers import multiclass_graph_with_external_predict


from .prediction_handlers import binary_fc_predict
from .prediction_handlers import regression_fc_predict
from .prediction_handlers import multiclass_fc_predict


def handle_prediction(model_data: dict, input_values: list):

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