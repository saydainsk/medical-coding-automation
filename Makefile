.PHONY: venv install run ui test fmt
venv:
	python -m venv .venv
install:
	. .venv/Scripts/activate && pip install -r requirements.txt
run:
	uvicorn services.coder_api.app:app --reload
ui:
	streamlit run ui/streamlit_review.py
test:
	python -m pytest -q
fmt:
	ruff check --fix || true && ruff format || true
