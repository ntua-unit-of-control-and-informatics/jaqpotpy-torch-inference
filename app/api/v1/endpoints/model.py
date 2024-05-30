from fastapi import APIRouter, Request

import uuid
from schemas.response import ModelUploadResponse
from schemas.predict import PredictRequest
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

    data = await req.json()

    data_for_db = {
        'files': data['files'],
        'task_params': data['task_params'],
        'metadata': data['metadata'],
    }

    # model_id = str(uuid.uuid4())
    model_id = 0

    db[model_id] = data_for_db

    task = data['metadata']['task']

    task_params = data['task_params']
    metadata = data['metadata']

    return ModelUploadResponse(model_id=model_id, message="Model uploaded successfully")





@router.post("/predict/", 
            #  response_model=ModelUploadResponse,
             summary="",
             description="")
async def predict(req: PredictRequest):

    model_id = req.model_id
    user_id = req.user_id
    user_input = req.user_input
    

    model_data = db[model_id] # will need await when a proper Database is used

    model_type = model_data['metadata']['model_type']


    outt = handle_prediction(model_data, user_input)



    # files = model_row['files']

    # model_scripted_base64 = files['model_scripted']
    # featurizer_pickle_base64 = files['featurizer_pickle']


    # model_scripted_content = base64.b64decode(model_scripted_base64)
    # featurizer_pickle_content = base64.b64decode(featurizer_pickle_base64)
    
    # model_buffer = io.BytesIO(model_scripted_content)
    # featurizer_buffer = io.BytesIO(featurizer_pickle_content)

    # model_buffer.seek(0)
    # featurizer_buffer.seek(0)

    # model = torch.jit.load(model_buffer)
    # featurizer = pickle.load(featurizer_buffer)
    

    # task_params = model_row['task_params']
    # metadata = model_row['metadata']

    # task = metadata['task']

    # print(task_params, metadata, task)

    return {'it is': 'ok'}



    




# @router.post("/{model_id}/")
# async def show_fields(request):
    