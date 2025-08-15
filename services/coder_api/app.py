from pathlib import Path
import os
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schemas import PredictRequest, PredictResponse, CodeCandidate
from nlp.preprocess import sectionize, is_negated, extract_rationales

load_dotenv()
APP_VERSION = "retrieval-baseline-0.2.1"

# ---------- Logging ----------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("medical-coding-api")

# ---------- Load codebook ----------
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "codebooks" / "icd10cm_codes.csv"
codes_df = pd.read_csv(DATA_PATH)

code_texts: List[str] = (
    codes_df["long_title"].fillna("") + " " + codes_df["short_title"].fillna("")
).tolist()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
code_matrix = vectorizer.fit_transform(code_texts)

# ---------- FastAPI app ----------
app = FastAPI(title="Medical Coding API", version=APP_VERSION)

# CORS: no credentials when using wildcard origins
origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,  # safer with "*" origins
)
# ---------- Core logic ----------
def retrieve_candidates(note: str, top_k: int) -> List[Tuple[int, float]]:
    secs = sectionize(note)
    weights: List[float] = []
    texts: List[str] = []
    for name, chunk in secs:
        if not chunk:
            continue
        w = 1.5 if name in {"assessment", "plan", "impression", "diagnosis"} else 1.0
        texts.append(chunk)
        weights.append(w)

    if not texts:
        texts = [note]
        weights = [1.0]

    sims = cosine_similarity(vectorizer.transform(texts), code_matrix)  # [S x C]
    sims = (np.array(weights)[:, None] * sims)  # section weighting
    scores = sims.max(axis=0)  # best-matching section per code

    # Negation suppression
    lowered = note.lower()
    penalties = np.ones_like(scores)
    for i, row in codes_df.iterrows():
        terms = [t for t in row["short_title"].lower().split() if len(t) > 3]
        if any(is_negated(t, lowered) for t in terms[:3]):
            penalties[i] = 0.3
    scores *= penalties

    order = np.argsort(-scores)
    return [(int(i), float(scores[i])) for i in order[:top_k] if scores[i] > 0]
# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"version": APP_VERSION}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    topk = max(1, min(req.top_k, 20))
    idx_scores = retrieve_candidates(req.note_text, topk)

    cands: List[CodeCandidate] = []
    for i, s in idx_scores:
        row = codes_df.iloc[i]
        key_terms = [w for w in row["short_title"].split() if len(w) > 3][:4]
        spans = extract_rationales(req.note_text, key_terms)
        cands.append(
            CodeCandidate(
                code=row["code"],
                title=row["long_title"],
                score=round(s, 4),
                rationale_spans=spans,
            )
        )

    log.info("predicted", extra={"top_k": topk, "num_candidates": len(cands)})
    return PredictResponse(candidates=cands, model_version=APP_VERSION)
