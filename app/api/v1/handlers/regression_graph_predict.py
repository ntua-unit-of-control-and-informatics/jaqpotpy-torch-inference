import base64
import io
import torch
import pickle
# import jaqpotpy


def regression_graph_predict(model_data, user_input):

    files = model_data['files']

    model_scripted_base64 = files['model_scripted']
    featurizer_pickle_base64 = files['featurizer_pickle']


    model_scripted_content = base64.b64decode(model_scripted_base64)
    featurizer_pickle_content = base64.b64decode(featurizer_pickle_base64)
    
    model_buffer = io.BytesIO(model_scripted_content)
    featurizer_buffer = io.BytesIO(featurizer_pickle_content)

    model_buffer.seek(0)
    featurizer_buffer.seek(0)

    model = torch.jit.load(model_buffer)
    featurizer = pickle.load(featurizer_buffer)

    sm = user_input['smiles']
    data_point = featurizer(sm)
    data_point.batch = torch.zeros(data_point.size(0), dtype=torch.int64)
    

    normalization_mean = model_data['task_params']['normalization_mean']
    normalization_std = model_data['task_params']['normalization_std']

    model.eval()
    with torch.no_grad():
        outputs = model(x=data_point.x, edge_index=data_point.edge_index, batch=data_point.batch).squeeze(-1)
        outputs.mul_(normalization_std).add_(normalization_mean)
    

    result = outputs[0].item()
    print(result)
    return 

    # task_params = model_data['task_params']
    # metadata = model_data['metadata']

    # task = metadata['task']

    # print(task_params, metadata, task)