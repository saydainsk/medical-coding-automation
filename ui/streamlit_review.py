# ui/streamlit_review.py
import os
import json
import time
import requests
from typing import Dict, List, Tuple
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from html import escape

load_dotenv()

DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8008/predict")
DEFAULT_API_KEY = os.getenv("API_KEY", "")
REGRESSION_CSV = os.getenv("REGRESSION_CSV", "data/eval/regression_prompts.csv")

st.set_page_config(page_title="Medical Coding UI", page_icon="💊", layout="wide")

# ---------- Sidebar: API settings ----------
with st.sidebar:
    st.header("API Settings")
    api_url = st.text_input(
        "Predict endpoint", value=DEFAULT_API_URL, help="FastAPI /predict URL"
    )
    api_key = st.text_input(
        "x-api-key (if set)", value=DEFAULT_API_KEY, type="password"
    )

    # Health check + latency
    if st.button("Check API health"):
        try:
            t0 = time.time()
            r = requests.get(api_url.replace("/predict", "/health"), timeout=5)
            ms = int((time.time() - t0) * 1000)
            if r.ok:
                st.success(f"Healthy • {ms} ms")
            else:
                st.error(f"HTTP {r.status_code}: {r.text}")
        except Exception as e:
            st.error(f"Health check failed: {e}")

    st.divider()

    # Client-side filtering & examples
    score_min = st.slider(
        "Min score (client-side filter)",
        0.0,
        1.0,
        0.20,
        0.01,
        help="Filter results in the UI without changing the API.",
    )
    top_k_sidebar = st.number_input(
        "Default Top-K",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Prefills the Top-K input below.",
    )

    # Load examples from regression CSV (optional)
    example_text = None
    if os.path.exists(REGRESSION_CSV):
        try:
            ex_df = pd.read_csv(REGRESSION_CSV)
            ex_id = st.selectbox("Load example prompt", ["--"] + ex_df["id"].tolist())
            if ex_id != "--":
                example_text = ex_df.loc[ex_df["id"] == ex_id, "note_text"].iloc[0]
                st.info("Loaded example into the editor.")
        except Exception as e:
            st.caption(f"Could not read {REGRESSION_CSV}: {e}")

    st.caption("Tip: set API_URL and API_KEY in your .env to pre-fill these.")
    st.markdown("[Open API docs](/docs) · [Metrics](/metrics)")

st.title("💊 Medical Coding — Demo UI")
st.write(
    "Enter a short clinical note. The app calls your FastAPI `/predict` and shows candidates, scores, and rationales."
)

# ---------- Note editor ----------
default_note = "Low back pain, denies pneumonia."
if "note_text" not in st.session_state:
    st.session_state["note_text"] = default_note
if example_text:
    st.session_state["note_text"] = example_text

note = st.text_area(
    "Clinical note",
    key="note_text",
    height=140,
    help="Free-text note to code.",
)

top_k = st.number_input(
    "Top-K", min_value=1, max_value=20, value=int(top_k_sidebar), step=1
)


# ---------- Helpers ----------
def call_api(note_text: str, top_k: int, url: str, key: str):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["x-api-key"] = key
    payload = {"note_text": note_text, "top_k": int(top_k)}
    t0 = time.time()
    r = requests.post(url, headers=headers, json=payload, timeout=20)
    latency_ms = int((time.time() - t0) * 1000)
    return r, latency_ms


def flatten_candidates(cands: List[Dict]) -> pd.DataFrame:
    rows = []
    for c in cands:
        rows.append(
            {
                "code": c.get("code", ""),
                "title": c.get("title", ""),
                "score": float(c.get("score", 0.0)),
                "rationales": json.dumps(c.get("rationale_spans", {})),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["score (%)"] = (df["score"] * 100).round(1).astype(str) + "%"
    return df


def spans_to_intervals(spans: Dict[str, List[List[int]]]) -> List[Tuple[int, int, str]]:
    intervals = []
    for term, arr in (spans or {}).items():
        for pair in arr or []:
            try:
                start, end = int(pair[0]), int(pair[1])
                if end > start >= 0:
                    intervals.append((start, end, term))
            except Exception:
                continue
    intervals.sort(key=lambda x: (x[0], x[1]))
    return intervals


def highlight_text(text: str, spans: Dict[str, List[List[int]]]) -> str:
    """Return HTML with <mark> around rationale spans."""
    if not text:
        return "<i>No text</i>"
    s = spans_to_intervals(spans)
    if not s:
        return f"<div style='white-space:pre-wrap'>{escape(text)}</div>"
    out = []
    i = 0
    for start, end, term in s:
        start = max(0, min(start, len(text)))
        end = max(0, min(end, len(text)))
        if start > i:
            out.append(escape(text[i:start]))
        seg = escape(text[start:end])
        out.append(f"<mark title='{escape(term)}'>{seg}</mark>")
        i = end
    if i < len(text):
        out.append(escape(text[i:]))
    return f"<div style='white-space:pre-wrap'>{''.join(out)}</div>"


# ---------- Layout ----------
col_left, col_right = st.columns([1.25, 1])

# ---------- Action ----------
if st.button("🔎 Predict", type="primary"):
    with st.spinner("Calling API..."):
        try:
            resp, latency_ms = call_api(note, top_k, api_url, api_key)
            rid = resp.headers.get("x-request-id")
            if resp.status_code != 200:
                st.error(f"HTTP {resp.status_code}: {resp.text}")
            else:
                data = resp.json()
                cands = data.get("candidates", [])
                model_version = data.get("model_version", "unknown")
                post_v = data.get("postprocess_version") or data.get("meta", {}).get(
                    "postprocess_version"
                )

                # Left: table + tools
                with col_left:
                    st.subheader("Candidates")
                    df = flatten_candidates(cands)

                    # client-side filter
                    if not df.empty:
                        df = df[df["score"] >= float(score_min)].reset_index(drop=True)

                    if not df.empty:
                        st.dataframe(
                            df[["code", "title", "score", "score (%)", "rationales"]],
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.download_button(
                            "⬇️ Download results (CSV)",
                            df.to_csv(index=False).encode("utf-8"),
                            file_name="candidates.csv",
                            mime="text/csv",
                        )
                    else:
                        st.info(
                            "No candidates returned (or filtered out by min score)."
                        )

                    st.caption(
                        f"model_version: `{model_version}`"
                        + (f" • postprocess_version: `{post_v}`" if post_v else "")
                        + (f" • x-request-id: `{rid}`" if rid else "")
                        + f" • latency: {latency_ms} ms"
                    )

                # Right: rationale + details
                with col_right:
                    st.subheader("Rationale preview (top-1)")
                    top1 = cands[0] if cands else None
                    if top1:
                        st.markdown(
                            highlight_text(note, top1.get("rationale_spans", {})),
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            f"Top-1: {top1.get('code')} • score={top1.get('score')}"
                        )
                    else:
                        st.write("—")

                    if not df.empty:
                        choice = st.selectbox(
                            "Details",
                            [
                                f"{r.code} – {r.title}"
                                for r in df.itertuples(index=False)
                            ],
                        )
                        if choice:
                            sel_code = choice.split(" – ", 1)[0]
                            sel_row = df[df["code"] == sel_code].iloc[0].to_dict()
                            with st.expander(
                                f"ICD-10-CM details for {sel_code}", expanded=True
                            ):
                                st.write(f"**Title:** {sel_row['title']}")
                                st.write(
                                    f"**Score:** {sel_row['score']:.4f} ({sel_row['score (%)']})"
                                )
                                st.write("**Rationales (raw):**")
                                st.code(sel_row["rationales"], language="json")

                # Expanders: raw + curl + JSON
                with st.expander("Raw response"):
                    st.code(json.dumps(data, indent=2), language="json")

                with st.expander("Copy as cURL"):
                    curl = (
                        f'curl -sS "{api_url}" '
                        f'-H "Content-Type: application/json" '
                        + (f'-H "x-api-key: {api_key}" ' if api_key else "")
                        + f"--data-raw {json.dumps({'note_text': note, 'top_k': int(top_k)})}"
                    )
                    st.code(curl, language="bash")

                with st.expander("Copy JSON body"):
                    st.code(
                        json.dumps({"note_text": note, "top_k": int(top_k)}, indent=2),
                        language="json",
                    )

                st.success("Done.")
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the API. Is the FastAPI server running and reachable?"
            )
        except Exception as e:
            st.exception(e)
else:
    with col_left:
        st.info(
            "Enter a note and click **Predict**. You can also load an example from the sidebar."
        )
