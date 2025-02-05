from fastapi import FastAPI
from mangum import Mangum

from app.api.endpoints import model

app = FastAPI()

app.include_router(model.router, prefix='/model', tags=['model'])

@app.get("/")
def read_root():
    return {"Hello": "Welcome to Dynalab 2.0"}

handler = Mangum(app)
