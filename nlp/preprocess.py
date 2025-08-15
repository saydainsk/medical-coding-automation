import regex as re
from typing import Dict, List, Tuple

SECTION_HEADERS = [
    "chief complaint", "history of present illness", "assessment", "plan",
    "impression", "diagnosis", "procedures", "past medical history"
]

NEG_PATTERNS = [
    r"\bno (?:evidence of|hx of|history of)?\s*{term}\b",
    r"\bdenies?\s+{term}\b",
    r"\bwithout\s+{term}\b",
]

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()

def sectionize(text: str):
    t = normalize(text)
    chunks = []
    last = 0
    pat = rf"({'|'.join(map(re.escape, SECTION_HEADERS))})[:\-]"
    for m in re.finditer(pat, t):
        if m.start() > last:
            chunks.append(("general", t[last:m.start()].strip()))
        chunks.append((m.group(1), ""))
        last = m.end()
    chunks.append(("general", t[last:].strip()))
    return chunks

def is_negated(term: str, context: str) -> bool:
    term_esc = re.escape(term.lower())
    for pat in NEG_PATTERNS:
        if re.search(pat.format(term=term_esc), context, flags=re.I):
            return True
    return False

def extract_rationales(note: str, keywords: List[str]) -> Dict[str, List[Tuple[int,int]]]:
    spans: Dict[str, List[Tuple[int,int]]] = {}
    low = note.lower()
    for kw in keywords:
        spans.setdefault(kw, [])
        for m in re.finditer(re.escape(kw.lower()), low):
            spans[kw].append((m.start(), m.end()))
    return spans
