# Amazon Reviews NLP Streamlit App

This repository contains a small Streamlit wrapper around `nlp_analysis.py` which performs:

- Named-Entity Recognition (spaCy `en_core_web_sm` model)
- Simple rule-based sentiment classification (Positive / Negative / Neutral)

Files added:

- `app.py` — Streamlit application to analyze single reviews or a compressed FastText `.bz2` file.
- `requirements.txt` — minimal Python dependencies.

Quick start (Windows PowerShell):

```powershell
# Create a venv
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Install spaCy model
python -m spacy download en_core_web_sm
# Run the app
streamlit run app.py
```

Notes:

- `app.py` imports `nlp_analysis.py` from the same folder and re-uses its `nlp`, `rule_based_sentiment`, and `extract_review_text` helpers.
- The Streamlit app limits text length sent to spaCy for performance.
