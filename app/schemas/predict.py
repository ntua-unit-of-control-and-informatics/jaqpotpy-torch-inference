from pydantic import BaseModel
from typing import List

# class PredictRequest(BaseModel):
#     model_id: int
#     user_id: int
#     user_input: dict


class SinglePredictionResult(BaseModel):
    """
    A model representing the result of a single prediction.

    Attributes:
    - id (int): The identifier for the prediction.
    - prediction (float): The predicted value.
    - prob (List[float], optional): A list of probabilities assigned to each class in the case of a classification task.
    - doa (bool, optional): A flag indicating if the prediction is associated with an input that belongs to the domain of applicability (DOA).
    """
    id: int
    prediction: float
    prob: List[float] = None
    doa: bool = None


class PredictResponse(BaseModel):
    """
    A model representing the response of a prediction request.

    Attributes:
    - results (List[SinglePredictionResult]): A list of results for each prediction made.
    - message (str, optional): A message associated with the prediction response.
    """
    results: List[SinglePredictionResult]
    message: str = None
    # other: Any = None