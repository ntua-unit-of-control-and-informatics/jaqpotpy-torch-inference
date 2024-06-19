import base64
import io
import torch
import pickle
import torch.nn.functional as F
from schemas import SinglePredictionResult


def multiclass_graph_predict(model_data: dict, user_inputs: list[dict]):
    
    model_scripted_base64 = model_data['actualModel']
    model_scripted_content = base64.b64decode(model_scripted_base64)
    model_buffer = io.BytesIO(model_scripted_content)
    model_buffer.seek(0)
    model = torch.jit.load(model_buffer)

    featurizer_pickle_base64 = model_data['additional_model_params']['featurizer']
    featurizer_pickle_content = base64.b64decode(featurizer_pickle_base64)
    featurizer_buffer = io.BytesIO(featurizer_pickle_content)
    featurizer_buffer.seek(0)
    featurizer = pickle.load(featurizer_buffer)

    results = []

    for i, user_input in enumerate(user_inputs):
        sm = user_input['SMILES']
        data_point = featurizer(sm)
        data_point.batch = torch.zeros(data_point.size(0), dtype=torch.int64)
        
        model.eval()
        with torch.no_grad():
            try:
                outputs = model(x=data_point.x, edge_index=data_point.edge_index, batch=data_point.batch, edge_attr=data_point.edge_attr)
            except RuntimeError: # if model doesn't support edge_attr (edge features)
                outputs = model(x=data_point.x, edge_index=data_point.edge_index, batch=data_point.batch)

            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

        result = SinglePredictionResult(
            id=i,
            prediction=preds[0].item(),
            prob=probs.squeeze(0).tolist()
        )
        results.append(result)

    return results
