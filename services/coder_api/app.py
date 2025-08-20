# ruff: noqa: E402
"""FastAPI app for medical coder API."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root is on sys.path before importing project-local packages
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from prometheus_fastapi_instrumentator import Instrumentator
from asgi_correlation_id import CorrelationIdMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schemas import CodeCandidate, PredictRequest, PredictResponse
from nlp.preprocess import (
    expand_abbrev,
    extract_rationales,
    is_negated,
    load_abbrev,
    sectionize,
)
from .postprocess_icd10 import postprocess_candidates

load_dotenv()
APP_VERSION = "retrieval-baseline-0.2.3"

# ---------- Logging ----------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("medical-coding-api")

# ---------- Load codebook ----------
from pathlib import Path

# --- Resolve codebook path (always end up with a Path) ---
_env = os.getenv("CODEBOOK_PATH", "").strip().strip('"').strip("'")
_default = (
    Path(__file__).resolve().parents[2] / "data" / "codebooks" / "icd10cm_codes.csv"
)

CODEBOOK_PATH: Path = Path(_env) if _env else _default

if not CODEBOOK_PATH.exists():
    raise FileNotFoundError(f"Codebook CSV not found at: {CODEBOOK_PATH}")

# Pylance-friendly call (str() is fine for pandas & avoids path-type warnings)
codes_df = pd.read_csv(str(CODEBOOK_PATH))


# Safely access optional columns (returns empty strings when column is absent)
def _safe(col: str) -> pd.Series:
    if col in codes_df.columns:
        return codes_df[col].fillna("").astype(str)
    # fallback: empty series of correct length
    return pd.Series([""] * len(codes_df), dtype="string")


# Enrich the retriever text with synonyms & includes_terms
code_texts: List[str] = (
    _safe("long_title")
    + " "
    + _safe("short_title")
    + " "
    + _safe("includes_terms")
    + " "
    + _safe("synonyms")
).tolist()

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
code_matrix = vectorizer.fit_transform(code_texts)

# Build a quick code→index lookup for remaps
_CODE_TO_IDX = {
    str(row.code): i for i, row in codes_df.reset_index(drop=True).iterrows()
}

# ---------- FastAPI app ----------
app = FastAPI(title="Medical Coding API", version=APP_VERSION)

# CORS: no credentials with wildcard origins
origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# Correlation ID middleware (adds request id to logs)
app.add_middleware(CorrelationIdMiddleware)

# Prometheus metrics at /metrics (not in OpenAPI schema)
Instrumentator().instrument(app).expose(app, include_in_schema=False)


# API key guard (optional; only enforced if API_KEY is set)
def require_api_key(x_api_key: str = Header(None, alias="x-api-key")):
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------- Core logic ----------
def retrieve_candidates(note: str, top_k: int) -> List[Tuple[int, float]]:
    # Expand abbreviations (LBP→low back pain, HTN→hypertension, etc.)
    note = expand_abbrev(note, load_abbrev(os.getenv("ABBREV_PATH")))
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
    sims = np.array(weights)[:, None] * sims  # section weighting
    scores = sims.max(axis=0)  # best-matching section per code

    # Negation suppression
    lowered = note.lower()
    penalties = np.ones_like(scores)
    for i, row in codes_df.iterrows():
        st = (row.get("short_title") or "").lower()
        terms = [t for t in st.split() if len(t) > 3]
        if any(is_negated(t, lowered) for t in terms[:3]):
            penalties[i] = 0.0
    scores *= penalties

    order = np.argsort(-scores)
    min_score = float(os.getenv("MIN_SCORE", "0.2"))
    return [(int(i), float(scores[i])) for i in order if scores[i] >= min_score][:top_k]


# ---------- Endpoints ----------
@app.get("/", include_in_schema=False)
def root():
    # Redirect to interactive docs
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # Avoid noisy 404s from browsers asking for a favicon
    return Response(status_code=204)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/live")
def live():
    return {"live": True}


@app.get("/ready")
def ready():
    try:
        _ = code_matrix.shape[0]
        return {"ready": True}
    except Exception:
        return {"ready": False}


@app.get("/version")
def version():
    return {"version": APP_VERSION}


@app.get("/codes")
def codes(limit: int = 5):
    lim = max(0, min(limit, 50))
    sample = codes_df.head(lim).to_dict(orient="records")
    return {"count": int(len(codes_df)), "sample": sample}


@app.post(
    "/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)]
)
def predict(req: PredictRequest):
    topk = max(1, min(req.top_k, 20))
    idx_scores = retrieve_candidates(req.note_text, topk)

    # Build a minimal candidate list (code + score) to feed the post-processor
    prelim = [
        {
            "code": str(codes_df.iloc[i]["code"]),
            "score": float(round(s, 4)),
        }
        for i, s in idx_scores
    ]

    # 🔗 Pass codes_df into the post-processor (this may remap codes & drop dups)
    processed = postprocess_candidates(prelim, req.note_text, codes_df)

    # Now rebuild the final response items with accurate titles & rationales
    cands: List[CodeCandidate] = []
    from nlp.preprocess import tokenize

    for c in processed:
        code = c["code"]
        score = float(c.get("score", 0.0))

        # Find the row for the (possibly remapped) code
        row_match = codes_df.loc[codes_df["code"] == code]
        if row_match.empty:
            # If a code isn't in the codebook (shouldn't happen), skip it safely
            continue
        row = row_match.iloc[0]

        short_title = row.get("short_title") or ""
        key_terms = [w for w in tokenize(short_title) if len(w) > 3][:4]
        spans = extract_rationales(req.note_text, key_terms)

        cands.append(
            CodeCandidate(
                code=code,
                title=row.get("long_title") or "",
                score=score,
                rationale_spans=spans,
            )
        )

        if len(cands) >= topk:
            break  # maintain top_k if postprocessing reduced/merged items

    log.info("predicted", extra={"top_k": topk, "num_candidates": len(cands)})
    return PredictResponse(
        candidates=cands,
        model_version=APP_VERSION,
        postprocess_version="icd10cm-remap-v1",
    )
