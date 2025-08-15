import pandas as pd


def test_codebook_parses_and_has_columns():
    df = pd.read_csv("data/codebooks/icd10cm_codes.csv")
    assert {"code", "short_title", "long_title"} <= set(df.columns)
    assert len(df) >= 3
