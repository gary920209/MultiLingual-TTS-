from pydantic import BaseModel, Field
from typing import List, Dict, Union
import numpy as np


class ModelSingleInput(BaseModel):
    """Input schema for single audio sample inference"""
    audio: List[float]
    sample_rate: int = Field(default=16000, gt=0)
    language: str
    
    def to_numpy(self) -> np.ndarray:
        """Convert audio to numpy array"""
        return np.array(self.audio, dtype=np.float32)

class ModelSingleOutput(BaseModel):
    language: str
    text: str