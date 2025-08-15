# Medical Coding Automation (Baseline)

[![CI](https://github.com/saydainsk/medical-coding-automation/actions/workflows/ci.yml/badge.svg)](https://github.com/saydainsk/medical-coding-automation/actions)

Paste a clinical note → get ranked ICD-10-CM suggestions with rationale spans.

## Quickstart
    python -m venv .venv && source .venv/Scripts/activate
    pip install -r requirements.txt
    uvicorn services.coder_api.app:app --reload

## Smoke test
    curl -s http://127.0.0.1:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"note_text":"Assessment: Type 2 diabetes and essential hypertension. Denies pneumonia. Low back pain.","top_k":5}' \
      | python -m json.tool

## Review UI
    streamlit run ui/streamlit_review.py

> CPT® is AMA-licensed. This baseline only demos ICD-10-CM.
