from pydantic import BaseModel
from typing import List, Dict, Tuple

class CodeCandidate(BaseModel):
    code: str
    title: str
    score: float
    rationale_spans: Dict[str, List[Tuple[int,int]]] = {}

class PredictRequest(BaseModel):
    note_text: str
    top_k: int = 5

class PredictResponse(BaseModel):
    candidates: List[CodeCandidate]
    model_version: str
