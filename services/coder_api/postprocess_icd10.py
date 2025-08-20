# services/coder_api/postprocess_icd10.py
from typing import List, Dict
import pandas as pd


def map_icd10(code: str, text: str) -> str:
    """
    Specific remaps (e.g., handle deprecated/non-billable parents).
    Extend this with more rules as needed.
    """
    c = (code or "").upper()
    tl = (text or "").lower()

    # Example: M54.5 (parent) → child codes (FY2025)
    if c == "M54.5":
        if "vertebrogenic" in tl:
            return "M54.51"
        return "M54.50"  # default child for unspecified

    return c


def _is_billable(code: str, codes_df: pd.DataFrame) -> bool:
    if "billable" in codes_df.columns:
        row = codes_df.loc[codes_df["code"] == code]
        if not row.empty:
            try:
                return bool(row.iloc[0]["billable"])
            except Exception:
                pass
    # Fallback heuristic: ICD-10-CM billable codes usually have ≥4 significant chars
    return len(code.replace(".", "")) >= 4


def _resolve_child(parent: str, codes_df: pd.DataFrame) -> str:
    """Pick a child of `parent` (prefer 'unspecified')."""
    children = codes_df[codes_df["code"].str.startswith(f"{parent}.")]
    if children.empty:
        return parent
    unspecified = children[
        children["long_title"].str.contains("unspecified", case=False, na=False)
    ]
    if not unspecified.empty:
        return str(unspecified.iloc[0]["code"])
    return str(children.iloc[0]["code"])


def postprocess_candidates(
    candidates: List[Dict], note_text: str, codes_df: pd.DataFrame
) -> List[Dict]:
    """
    Apply rule-based remaps (map_icd10), ensure billable codes, then dedupe.
    `candidates` items must at least have {"code": ..., "score": ...}.
    """
    seen, out = set(), []
    for c in candidates:
        code = map_icd10(c["code"], note_text)
        if not _is_billable(code, codes_df):
            code = _resolve_child(code, codes_df)
        if code in seen:
            continue
        seen.add(code)
        out.append({**c, "code": code})
    return out


__all__ = ["map_icd10", "postprocess_candidates"]
