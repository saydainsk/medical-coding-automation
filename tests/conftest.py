import os
from pathlib import Path
import pytest


@pytest.fixture(autouse=True, scope="session")
def set_codebook_env():
    # Don’t rely on legacy names; default to the enriched sample.
    default = "data/codebooks/icd10cm_codes_enriched_sample.csv"
    os.environ.setdefault("CODEBOOK_PATH", default)
    assert Path(
        os.environ["CODEBOOK_PATH"]
    ).exists(), f"Missing codebook at {os.environ['CODEBOOK_PATH']}"
