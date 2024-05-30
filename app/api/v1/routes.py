from fastapi import APIRouter
from .endpoints import model

router = APIRouter()


router.include_router(model.router, prefix="/models")
