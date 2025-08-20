# services/coder_api/schemas.py
from typing import List, Tuple
from pydantic import BaseModel, Field


class CodeCandidate(BaseModel):
    code: str
    title: str
    score: float
    rationale_spans: List[Tuple[int, int]] = Field(default_factory=list)


class PredictRequest(BaseModel):
    note_text: str
    top_k: int = 5


class PredictResponse(BaseModel):
    candidates: List[CodeCandidate]
    model_version: str
    postprocess_version: str
