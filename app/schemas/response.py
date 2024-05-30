from pydantic import BaseModel

class ModelUploadResponse(BaseModel):
    model_id: int
    message: str

        # protected_namespaces = ()
