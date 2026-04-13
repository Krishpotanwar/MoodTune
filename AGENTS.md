# MoodTune — Codex Instructions

## Project
Mood-based music recommender. Python + Streamlit. College evaluation project.
No audio playback. No paid APIs. Dataset: Kaggle Spotify CSV.

## Stack
- Python 3.11
- pandas, numpy, scikit-learn, streamlit, plotly, seaborn
- No LangChain, no OpenAI, no external APIs

## Commands
- Run app: streamlit run ui/app.py
- Validate data: python src/validator.py
- Install deps: pip install -r requirements.txt

## Structure
src/ → core logic modules (validator, mood_mapper, recommender, visualizer)
ui/  → streamlit app + CSS
data/raw/ → original CSV
data/processed/ → cleaned output
logs/ → validation_summary.txt

## Rules
- All functions must have type hints and docstrings
- Never hardcode file paths — use pathlib.Path from config
- Streamlit UI must use st.session_state for multi-step survey (no HTML forms)
- Custom CSS goes in ui/styles.css and is injected via st.markdown in app.py
- validator.py must write logs/validation_summary.txt on every run
- Keep modules independent — recommender.py must not import from ui/

## Coding style
- PEP8, snake_case, modular functions under 40 lines each
- Prefer explicit over implicit — no magic numbers, use named constants

## Verification
After each module: run it standalone with python src/<module>.py and confirm no errors.
After UI work: run streamlit run ui/app.py and visually verify the step renders.