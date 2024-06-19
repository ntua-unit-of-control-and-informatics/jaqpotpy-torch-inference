import base64
import io
import torch
import pickle
import torch.nn.functional as F
import pandas as pd


def binary_fc_predict(model_data: dict, user_inputs: list[dict]):

    model_scripted_base64 = model_data['actualModel']
    model_scripted_content = base64.b64decode(model_scripted_base64)
    model_buffer = io.BytesIO(model_scripted_content)
    model_buffer.seek(0)
    model = torch.jit.load(model_buffer)

    preprocessor_pickle_base64 = model_data['additional_model_params']['preprocessor']
    preprocessor_pickle_content = base64.b64decode(preprocessor_pickle_base64)
    preprocessor_buffer = io.BytesIO(preprocessor_pickle_content)
    preprocessor_buffer.seek(0)
    preprocessor = pickle.load(preprocessor_buffer)

    decision_threshold = model_data['additional_model_params']['decision_threshold']

    df = pd.DataFrame(user_inputs)
    df = preprocessor.transform(df)
    if isinstance(df, pd.DataFrame):
         df = df.to_numpy()
    df = torch.tensor(df).float()

    results = []

    for i, x in enumerate(df):
        x = x.unsqueeze(0)
        model.eval()
        with torch.no_grad():
            outputs = model(x).squeeze(-1)
            probs = F.sigmoid(outputs)
            preds = (probs > decision_threshold).int()

        result = {
            'id': i,
            'prediction': preds[0].item(),
            'prob': [probs[0].item(), 1 - probs[0].item()]
        }
        results.append(result)
        
    return results
