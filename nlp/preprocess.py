# nlp/preprocess.py
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import regex as re

# ---------------------- Config / Lexicons ----------------------
SECTION_HEADERS = [
    "chief complaint",
    "history of present illness",
    "assessment",
    "plan",
    "impression",
    "diagnosis",
    "procedures",
    "past medical history",
    "hpi",
    "ros",
    "exam",
    "physical exam",
    "pe",
]

NEG_PATTERNS = [
    r"\bno (?:evidence of\s+)?{term}\b",
    r"\bno hx of\s+{term}\b",
    r"\bno history of\s+{term}\b",
    r"\bdenies?\s+{term}\b",
    r"\bwithout\s+{term}\b",
    r"\bnegative for\s+{term}\b",
]

WORD_RE = re.compile(r"[a-z0-9]+", re.I)

# Abbreviation cache (loaded on first use)
_ABBREV_DF: pd.DataFrame | None = None
_ABBREV_REGEX: re.Pattern | None = None
_ABBREV_MAP: Dict[str, str] = {}


def load_abbrev(path: str | None = None) -> pd.DataFrame:
    """
    Load abbreviation lexicon (term, expansion[, weight]).
    Default path: data/lexicons/abbrev.csv or env ABBREV_PATH.
    """
    global _ABBREV_DF, _ABBREV_REGEX, _ABBREV_MAP

    if _ABBREV_DF is not None:
        return _ABBREV_DF

    # Resolve path
    candidates = []
    if path:
        candidates.append(Path(path))
    env_p = os.getenv("ABBREV_PATH")
    if env_p:
        candidates.append(Path(env_p))
    # repo-relative defaults
    here = Path(__file__).resolve()
    candidates.append(here.parents[2] / "data" / "lexicons" / "abbrev.csv")
    candidates.append(Path("data/lexicons/abbrev.csv"))

    src = next((p for p in candidates if p and p.exists()), None)
    if src and src.exists():
        df = pd.read_csv(src)
    else:
        df = pd.DataFrame(columns=["term", "expansion", "weight"])

    # Build fast lookup + regex
    terms = []
    amap: Dict[str, str] = {}
    for _, r in df.iterrows():
        t = str(r.get("term", "")).strip()
        e = str(r.get("expansion", "")).strip()
        if not t or not e:
            continue
        terms.append(re.escape(t))
        amap[t.lower()] = e

    if terms:
        patt = r"\b(?:%s)\b" % "|".join(sorted(terms, key=len, reverse=True))
        _ABBREV_REGEX = re.compile(patt, flags=re.IGNORECASE)
    else:
        _ABBREV_REGEX = None

    _ABBREV_DF = df
    _ABBREV_MAP = amap
    return _ABBREV_DF


def expand_abbrev(text: str, df: pd.DataFrame | None = None) -> str:
    """
    Expand abbreviations using the loaded lexicon.
    Returns the original text PLUS an expanded copy to enrich retrieval:
      "LBP improving" -> "LBP improving low back pain improving"
    """
    global _ABBREV_REGEX, _ABBREV_MAP
    if df is None:
        df = load_abbrev()

    if _ABBREV_REGEX is None or df.empty:
        return text

    def _repl(m: re.Match) -> str:
        key = m.group(0).lower()
        return _ABBREV_MAP.get(key, m.group(0))

    expanded = _ABBREV_REGEX.sub(_repl, text)
    if expanded == text:
        return text  # nothing changed
    return f"{text} {expanded}"


# ---------------------- Core helpers ----------------------
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def sectionize(text: str) -> List[Tuple[str, str]]:
    """
    Split the note into (section_name, chunk) pairs.
    Looks for known headers at line starts like 'Assessment:' or 'Plan -'.
    """
    # Work on the original text (not fully normalized) to preserve spans
    headers = [re.escape(h) for h in SECTION_HEADERS]
    pat = re.compile(rf"^\s*({'|'.join(headers)})\s*[:\-]\s*", flags=re.I | re.M)

    sections: List[Tuple[str, str]] = []
    cur_name = "general"
    last = 0

    for m in pat.finditer(text):
        start, end = m.span()
        # previous chunk
        chunk = text[last:start].strip()
        if chunk:
            sections.append((cur_name.lower(), chunk))
        # switch to new section
        cur_name = m.group(1).lower()
        last = end

    # tail
    tail = text[last:].strip()
    if tail:
        sections.append((cur_name.lower(), tail))

    # If nothing matched at all, return a single general chunk
    if not sections:
        return [("general", text.strip())]
    return sections


def is_negated(term: str, context: str) -> bool:
    term_esc = re.escape(term.lower())
    for pat in NEG_PATTERNS:
        if re.search(pat.format(term=term_esc), context, flags=re.I):
            return True
    return False


def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def extract_rationales(
    note: str, keywords: List[str]
) -> Dict[str, List[Tuple[int, int]]]:
    """Return spans (start,end) for each keyword (case-insensitive, word-boundary)."""
    spans: Dict[str, List[Tuple[int, int]]] = {}
    low = note.lower()
    for kw in keywords:
        kw_clean = kw.lower().strip()
        if not kw_clean:
            continue
        spans.setdefault(kw, [])
        for m in re.finditer(rf"\b{re.escape(kw_clean)}\b", low, flags=re.I):
            spans[kw].append((m.start(), m.end()))
    return spans
