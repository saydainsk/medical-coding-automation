# services/coder_api/codebook.py
from __future__ import annotations
import os
import pathlib
import pandas as pd
from functools import lru_cache

# Single source of truth for the path
CODEBOOK_PATH = pathlib.Path(
    os.getenv("ICD10_CODES_CSV", "data/codebooks/icd10cm_codes_enriched_sample.csv")
)


@lru_cache(maxsize=1)
def load_codebook() -> pd.DataFrame:
    if not CODEBOOK_PATH.exists():
        raise FileNotFoundError(
            f"ICD-10 codebook not found at {CODEBOOK_PATH}. "
            "Set $ICD10_CODES_CSV to the correct CSV path."
        )
    df = pd.read_csv(CODEBOOK_PATH)
    required = {"code", "short_title", "long_title"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Codebook missing columns: {missing}. Found: {list(df.columns)}"
        )
    return df
