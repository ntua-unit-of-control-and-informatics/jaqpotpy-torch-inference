import uvicorn

from fastapi import FastAPI
from api.v1.routes import router


app = FastAPI()
app.include_router(router, prefix="/api/v1")


@app.get("/")
def read_root():
    return {"app": "jaqpotpy-torch-inference",
            "message": "Welcome to Jaqpotpy Torch Inference module!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006, reload=False)
