from fastapi import APIRouter
from app.domain.model import ModelController
from app.domain.schemas.model import ModelSingleInput, ModelSingleOutput
from typing import List 

router = APIRouter()


@router.post("/single_evaluation", response_model=ModelSingleOutput)
async def single_evaluation(data: ModelSingleInput):
    model = ModelController()
    answer = model.single_inference(data)
    return answer