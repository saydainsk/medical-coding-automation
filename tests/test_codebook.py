import os
from pathlib import Path
import pandas as pd


def test_codebook_parses_and_has_columns():
    path = Path(
        os.getenv("CODEBOOK_PATH", "data/codebooks/icd10cm_codes_enriched_sample.csv")
    )
    assert path.exists(), f"Missing codebook at {path}"
    df = pd.read_csv(path)
    assert {"code", "short_title", "long_title"} <= set(df.columns)
