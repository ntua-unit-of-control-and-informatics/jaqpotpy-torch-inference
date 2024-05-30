from fastapi import HTTPException

from .regression_graph_predict import regression_graph_predict
from .binary_graph_predict import binary_graph_predict
# from .multiclass_graph_predict import multiclass_graph_predict




def handle_prediction(model_data: dict, user_input: dict):

    model_type = model_data['metadata']['model_type']

    match model_type:
        case "regression-graph-model":
            return regression_graph_predict(model_data, user_input)
        case "binary-graph-model":
            return binary_graph_predict(model_data, user_input)
        # case "multiclass-graph-model":
        #     return multiclass_graph_predict(model_data, user_input)
        case _:
            raise HTTPException(status_code=404, detail=f"Can't make prediction of model with model_type {model_type}.")