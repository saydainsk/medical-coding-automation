from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

import pandas as pd


class CandidateDict(TypedDict):
    code: str
    title: str
    score: float


def load_default_codebook() -> pd.DataFrame:
    """
    Load an ICD-10 codebook using CODEBOOK_PATH or the enriched sample.
    We DO NOT fall back to the legacy icd10cm_codes.csv to avoid CI/file drift.
    """
    root = Path(__file__).resolve().parents[2]
    env = os.getenv("CODEBOOK_PATH", "").strip().strip('"').strip("'")

    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    candidates.append(root / "data" / "codebooks" / "icd10cm_codes_enriched_sample.csv")

    for p in candidates:
        if p.exists():
            return pd.read_csv(str(p))

    raise FileNotFoundError(
        "Could not locate ICD-10 codebook via CODEBOOK_PATH or enriched sample."
    )


def _is_billable(code: str, codes_df: pd.DataFrame) -> bool:
    """Return True if the code is billable, using the column if present; else a heuristic."""
    if "billable" in codes_df.columns:
        row = codes_df.loc[codes_df["code"] == code]
        if not row.empty:
            try:
                return bool(row.iloc[0]["billable"])
            except Exception:
                pass  # fall through to heuristic
    # Heuristic: billable ICD-10-CM codes usually have ≥ 4 significant chars (excluding dot)
    return len(code.replace(".", "")) >= 4


def _resolve_child(parent: str, codes_df: pd.DataFrame) -> str:
    """
    Pick a child of `parent` (prefer 'unspecified' child if present).
    If none found, return the parent unchanged.
    """
    codes = codes_df["code"].astype(str)
    children = codes_df[codes.str.startswith(f"{parent}.", na=False)]
    if children.empty:
        return parent

    lt = children.get("long_title")
    if lt is not None:
        unspecified = children[
            lt.astype(str).str.contains("unspecified", case=False, na=False)
        ]
        if not unspecified.empty:
            return str(unspecified.iloc[0]["code"])

    return str(children.iloc[0]["code"])


def map_icd10(code: str, text: str) -> str:
    """
    Minimal rule-based remaps for tests/common cases.
    Handles M54.5 -> M54.50 (unspecified) or M54.51 (vertebrogenic).
    """
    code = str(code)
    t = (text or "").lower()

    if code == "M54.5":
        if "vertebrogenic" in t:
            return "M54.51"  # Vertebrogenic low back pain
        return "M54.50"  # Low back pain, unspecified

    return code


def postprocess_candidates(
    candidates: list[CandidateDict],
    note_text: str,
    codes_df: pd.DataFrame | None = None,  # <-- optional to satisfy tests
) -> list[CandidateDict]:
    """
    Backward-compatible postprocess:
      - If codes_df is None (older call sites/tests), load a default.
      - Apply rule-based remaps (map_icd10).
      - If result is non-billable/parent, resolve to an appropriate child.
      - Refresh titles from codebook when available.
      - Dedupe by code, keeping the highest score.
    """
    if codes_df is None:
        codes_df = load_default_codebook()

    note_low = (note_text or "").lower()
    remapped: list[CandidateDict] = []

    for c in candidates:
        # Normalize inputs
        in_code = str(c.get("code", ""))
        score = float(c.get("score", 0.0))
        title = c.get("title", "")

        # Rule-based remap (e.g., M54.5 -> M54.50/M54.51)
        code = map_icd10(in_code, note_low)

        # Ensure billable: if not, pick a sensible child (e.g., unspecified)
        if not _is_billable(code, codes_df):
            code = _resolve_child(code, codes_df)

        # Refresh title from codebook if available
        row = codes_df.loc[codes_df["code"] == code]
        if not row.empty:
            r0 = row.iloc[0]
            title = r0.get("long_title") or r0.get("short_title") or title

        remapped.append(CandidateDict(code=code, title=str(title), score=score))

    # Deduplicate by code, keeping the highest score
    best: dict[str, CandidateDict] = {}
    for item in remapped:
        k = item["code"]
        if k not in best or item["score"] > best[k]["score"]:
            best[k] = item

    return list(best.values())


__all__ = ["map_icd10", "postprocess_candidates"]
