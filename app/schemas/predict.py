from pydantic import BaseModel

class PredictRequest(BaseModel):
    model_id: int
    user_id: int
    user_input: dict