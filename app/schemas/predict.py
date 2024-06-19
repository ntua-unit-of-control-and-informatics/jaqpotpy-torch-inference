from pydantic import BaseModel
from typing import List

# class PredictRequest(BaseModel):
#     model_id: int
#     user_id: int
#     user_input: dict


class SinglePredictionResult(BaseModel):
    id: int
    prediction: float
    prob: List[float] = None
    doa: bool = None


class PredictResponse(BaseModel):
    results: List[SinglePredictionResult]
    message: str = None
    # other: Any = None