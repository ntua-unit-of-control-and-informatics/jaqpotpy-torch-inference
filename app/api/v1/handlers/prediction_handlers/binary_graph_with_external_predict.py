import base64
import io
import torch
import pickle
import torch.nn.functional as F
import pandas as pd
from schemas import SinglePredictionResult


def binary_graph_with_external_predict(model_data: dict, user_inputs: list[dict]):
    """
    Perform binary classification using a graph-based model combined with a fully-connected network
    making use of both SMILES and external features.

    Args:
    - model_data (dict): A dictionary containing model information and parameters.
        - 'actualModel' (str): Base64-encoded representation of the model script.
        - 'additional_model_params' (dict): Additional model parameters.
            - 'featurizer' (str): Base64-encoded representation of the featurizer used for input.
            - 'external_preprocessor' (str): Base64-encoded representation of the external data preprocessor.
            - 'decision_threshold' (float): Decision threshold for classification.
    - user_inputs (list[dict]): List of user inputs, where each input is a dictionary containing at least 'SMILES'.

    Returns:
    - list[SinglePredictionResult]: A list of prediction results for each user input.
    """

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

    external_preprocessor_pickle_base64 = model_data['additional_model_params']['external_preprocessor']
    external_preprocessor_pickle_content = base64.b64decode(external_preprocessor_pickle_base64)
    external_preprocessor_buffer = io.BytesIO(external_preprocessor_pickle_content)
    external_preprocessor_buffer.seek(0)
    external_preprocessor = pickle.load(external_preprocessor_buffer)

    decision_threshold = model_data['additional_model_params']['decision_threshold']

    df = pd.DataFrame(user_inputs)
    smiles = df['SMILES'].tolist()
    df = df.drop(columns=['SMILES'])

    df = external_preprocessor.transform(df)
    if isinstance(df, pd.DataFrame):
         df = df.to_numpy()
    df = torch.tensor(df).float()

    results = []

    for i, (sm, external) in enumerate(zip(smiles, df)):
        data_point = featurizer(sm)
        data_point.batch = torch.zeros(data_point.size(0), dtype=torch.int64)

        external = external.unsqueeze(0)

        model.eval()
        with torch.no_grad():
            try:
                outputs = model(x=data_point.x, edge_index=data_point.edge_index, external=external, batch=data_point.batch, edge_attr=data_point.edge_attr)
            except RuntimeError: # if model doesn't support edge_attr (edge features)
                outputs = model(x=data_point.x, edge_index=data_point.edge_index, external=external, batch=data_point.batch)
            
            outputs = outputs.squeeze(-1)
            probs = F.sigmoid(outputs)
            preds = (probs > decision_threshold).int()

        result = SinglePredictionResult(
            id=i, 
            prediction=preds[0].item(), 
            prob=[1 - probs[0].item(), probs[0].item()]
        )
        results.append(result)
        
    return results
