from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Tuple


class CodeCandidate(BaseModel):
    code: str
    title: str
    score: float
    rationale_spans: Dict[str, List[Tuple[int, int]]] = {}


class PredictRequest(BaseModel):
    note_text: str = Field(min_length=1, description="Clinical note text")
    top_k: int = Field(5, ge=1, le=20, description="How many codes to return (1..20)")

    @field_validator("note_text")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if v is None or not str(v).strip():
            raise ValueError("note_text is empty")
        return v


class PredictResponse(BaseModel):
    candidates: List[CodeCandidate]
    model_version: str
