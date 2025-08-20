# services/coder_api/schemas.py
from typing import List, Optional, Dict
from pydantic import BaseModel


class PredictRequest(BaseModel):
    note_text: str
    top_k: int = 5


class CodeCandidate(BaseModel):
    code: str
    title: str
    score: float
    rationale_spans: Dict[str, List[List[int]]]  # or just dict if you prefer


class PredictResponse(BaseModel):
    candidates: List[CodeCandidate]
    model_version: str
    # provenance/meta
    postprocess_version: Optional[str] = None
    # If you chose a meta dict instead of the field above, use:
    # meta: Dict[str, Any] = {}
