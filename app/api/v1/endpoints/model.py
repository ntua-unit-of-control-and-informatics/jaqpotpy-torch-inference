from fastapi import APIRouter, Request

import uuid
from schemas import ModelUploadResponse
from schemas import PredictResponse
import base64
import io
import torch
import pickle

from ..handlers.prediction_handler import handle_prediction



import sys
jaqpotpy_path = "../../../PINK-jaqpotpy/jaqpotpy"
running_locally = True
if jaqpotpy_path not in sys.path and running_locally:
    sys.path.append("../../../PINK-jaqpotpy/jaqpotpy")
import jaqpotpy




router = APIRouter(tags=["Models"])

db = {}


@router.get("/")
def read_root():
    return {"app": "jaqpotpy-torch-inference",
            "route": "/models"}


@router.get("/db/")
def get_db():
    return {'db': db}


@router.post("/upload/", 
             response_model=ModelUploadResponse,
             summary="Model Upload",
             description="Endpoint to upload a PyTorch Model and store it in the database.")
async def model_upload(req: Request):

    model_data = await req.json()

    # data_for_db = {
    #     'files': data['files'],
    #     'task_params': data['task_params'],
    #     'metadata': data['metadata'],
    # }
    model_data['id'] = len(db.keys())
    model_id = model_data['id']
    db[model_id] = model_data

    return ModelUploadResponse(model_id=model_id, message="Model uploaded successfully")





@router.post("/predict/", 
             response_model=PredictResponse,
             summary="",
             description="Endpoint to make predictions using a PyTorch model")
async def predict(req: Request):

    request_data = await req.json()


    # model_data = request_data['model'] # 
    model_id = request_data['model']['id']
    model_data = db[model_id]  # will need await when a proper Database is used

    # user_id = request_data['user_id']
    dataset = request_data['dataset']
    dataset_type = dataset['type']

    input_type = dataset['input']['type']
    input_values = dataset['input']['values']

    results = handle_prediction(model_data, input_values)


    return PredictResponse(results=results, message="Successful Prediction")


    




# @router.post("/{model_id}/")
# async def show_fields(request):
    