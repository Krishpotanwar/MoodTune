# SPEC.md — MoodTune: Mood-Based Music Recommender

**Version:** 1.0  
**Status:** DRAFT — awaiting author confirmation  
**Author:** Krish Potanwar  
**Stack:** Python 3.11, pandas, numpy, scikit-learn, streamlit, plotly, seaborn

---

## 1. Project Overview

MoodTune is a college project demonstrating an end-to-end ML pipeline:  
raw CSV → validated/cleaned dataset → cosine-similarity recommender → Streamlit UI.

The system asks the user 4 mood survey questions, maps answers to a 4-dimensional audio feature vector, and returns the top 8 closest tracks from a Spotify dataset using `sklearn.neighbors.NearestNeighbors`.

**No paid APIs. No audio playback. No LLMs.**

---

## 2. Dataset

| Property | Value |
|----------|-------|
| File | `SpotifyFeatures.csv` |
| Source | Kaggle — *zaheenhamidani/ultimate-spotify-tracks-db* |
| Rows | ~232,725 tracks |
| License | CC0 (public domain) |
| Placement | `data/raw/SpotifyFeatures.csv` |

### Required columns used

| Column | Type | Role |
|--------|------|------|
| `track_name` | str | Display on result card |
| `artist_name` | str | Display on result card |
| `genre` | str | Genre filter |
| `album` | str | Display on result card (if present) |
| `energy` | float [0–1] | Feature vector dim 1 |
| `valence` | float [0–1] | Feature vector dim 2 |
| `tempo` | float [0–250] | Feature vector dim 3 (normalised) |
| `acousticness` | float [0–1] | Feature vector dim 4 |

> `tempo` is min-max normalised to [0, 1] during cleaning and stored as `tempo_norm` in the processed CSV.

---

## 3. Mood Quadrants

The four mood categories used throughout the system:

| Quadrant | Energy | Valence | Typical feel |
|----------|--------|---------|-------------|
| **Joyful** | high (≥ 0.5) | high (≥ 0.5) | upbeat, celebratory |
| **Angry** | high (≥ 0.5) | low (< 0.5) | intense, aggressive |
| **Relaxed** | low (< 0.5) | high (≥ 0.5) | calm, peaceful |
| **Depressed** | low (< 0.5) | low (< 0.5) | melancholic, somber |

Quadrant is determined from the user's survey energy and valence scores.  
It is shown as a badge on the results screen.

---

## 4. Mood Survey — 4 Questions

Survey is multi-step (one question per screen) using `st.session_state`.  
A progress bar advances 25 % per step.

### Q1 — Energy
> *"How energetic do you feel right now?"*

| Option | Label | `energy` score |
|--------|-------|---------------|
| 1 | Exhausted | 0.10 |
| 2 | Low | 0.35 |
| 3 | Moderate | 0.65 |
| 4 | Pumped | 0.90 |

### Q2 — Positivity
> *"How positive is your mood?"*

| Option | Label | `valence` score |
|--------|-------|----------------|
| 1 | Very negative | 0.10 |
| 2 | Low | 0.35 |
| 3 | Neutral | 0.60 |
| 4 | Great | 0.90 |

### Q3 — Activity
> *"What are you currently doing?"*

| Option | Label | `tempo_norm` score |
|--------|-------|-------------------|
| 1 | Resting | 0.15 |
| 2 | Working / Studying | 0.40 |
| 3 | Working out | 0.70 |
| 4 | Partying | 0.90 |

### Q4 — Social setting
> *"Are you alone or with others?"*

| Option | Label | `acousticness` score |
|--------|-------|---------------------|
| 1 | Alone | 0.85 |
| 2 | With 1–2 friends | 0.55 |
| 3 | Small group | 0.30 |
| 4 | Big crowd | 0.10 |

### Resulting target vector

```python
target_vector = [energy, valence, tempo_norm, acousticness]
# Example: Pumped + Great + Partying + Big crowd → [0.90, 0.90, 0.90, 0.10]
```

---

## 5. Recommendation Engine

**Algorithm:** `sklearn.neighbors.NearestNeighbors`  
**Metric:** cosine  
**k:** 8 (top 8 tracks returned)  
**Input features:** `[energy, valence, tempo_norm, acousticness]`  
**Data scope:** full cleaned CSV (all genres, unless genre filter active)

### Optional genre filter

- Shown as a `st.multiselect` on the results page (below the mood badge).
- Selecting one or more genres re-runs `NearestNeighbors` on the filtered subset.
- Selecting nothing = search across all genres.

### Similarity score

Cosine similarity is computed as `1 - cosine_distance` and displayed as a percentage on each result card (e.g., **94% match**).

---

## 6. Module Specifications

### `src/validator.py`

```
Purpose : Data cleaning pipeline for SpotifyFeatures.csv
Input   : data/raw/SpotifyFeatures.csv
Output  : data/processed/SpotifyFeatures_clean.csv
          logs/validation_summary.txt
```

**Steps (in order):**

1. Load CSV, log raw shape and column dtypes.
2. Drop exact duplicate rows (keep first).
3. Drop rows where any of the 4 feature columns are null.
4. Clip feature values to valid ranges:
   - `energy`, `valence`, `acousticness` → clip to [0.0, 1.0]
   - `tempo` → clip to [0.0, 250.0]
5. Min-max normalise `tempo` → `tempo_norm` column.
6. Strip leading/trailing whitespace from `track_name`, `artist_name`, `genre`.
7. Save cleaned CSV to `data/processed/`.
8. Write a human-readable summary to `logs/validation_summary.txt`.

**Validation summary format:**

```
=== MoodTune Validation Report ===
Run at         : 2026-04-12 18:30:00
Raw rows       : 232725
Duplicates removed : 1847
Null rows removed  : 312
Clip adjustments   : 0 (energy), 2 (valence), 0 (acousticness), 14 (tempo)
Clean rows     : 230566
Genres found   : 26
Tempo range    : 0.0 – 244.95 BPM
=================================
```

**Public API:**

```python
def run_pipeline(raw_path: str, output_path: str, log_path: str) -> pd.DataFrame: ...
def get_cleaning_steps_log() -> list[dict]: ...   # for Data Lab UI
```

`get_cleaning_steps_log()` returns a list of step dictionaries used by the Data Lab tab:

```python
[
  {"step": 1, "name": "Load raw CSV",        "before": 232725, "after": 232725, "removed": 0},
  {"step": 2, "name": "Remove duplicates",   "before": 232725, "after": 230878, "removed": 1847},
  ...
]
```

---

### `src/mood_mapper.py`

```
Purpose : Convert survey answers (1–4 per question) to a feature vector + mood label
```

**Public API:**

```python
def map_to_vector(q1: int, q2: int, q3: int, q4: int) -> dict:
    """
    Returns:
        {
          "energy": float,
          "valence": float,
          "tempo_norm": float,
          "acousticness": float,
          "vector": list[float],       # ordered [energy, valence, tempo_norm, acousticness]
          "mood_label": str,           # "Joyful" | "Angry" | "Relaxed" | "Depressed"
          "mood_emoji": str            # "😄" | "😤" | "😌" | "😔"
        }
    """
```

Mapping tables are defined as module-level constants (no magic numbers inside functions).

---

### `src/recommender.py`

```
Purpose : Build NearestNeighbors model from cleaned CSV, return top-k results
```

**Public API:**

```python
def build_model(df: pd.DataFrame) -> NearestNeighbors: ...

def recommend(
    model: NearestNeighbors,
    df: pd.DataFrame,
    target_vector: list[float],
    k: int = 8,
    genre_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      track_name, artist_name, album, genre,
      energy, valence, tempo_norm, acousticness,
      similarity_pct   (float, 0–100)
    Ordered by similarity_pct descending.
    """
```

If `genre_filter` is non-empty, fit a fresh model on the filtered subset before querying.

---

### `src/visualizer.py`

```
Purpose : Generate the 3 required charts
```

**Chart 1 — Grouped bar: User mood vs recommended average**

```python
def mood_vs_recommended_bar(
    user_vector: dict,
    recommendations: pd.DataFrame,
) -> plotly.graph_objects.Figure: ...
```

- X-axis: feature names (Energy, Valence, Tempo, Acousticness)
- Two bar groups: "Your Mood" (blue) and "Avg Recommended" (purple)
- Dark background matching UI theme

**Chart 2 — Correlation heatmap (seaborn)**

```python
def feature_correlation_heatmap(df: pd.DataFrame) -> matplotlib.figure.Figure: ...
```

- Computed on the cleaned dataset's 4 feature columns + danceability (if present)
- Uses `coolwarm` colormap, annotated cells, dark background

**Chart 3 — Scatter: valence vs energy coloured by mood quadrant**

```python
def mood_cluster_scatter(
    df: pd.DataFrame,
    user_vector: dict,
    recommendations: pd.DataFrame,
) -> plotly.graph_objects.Figure: ...
```

- Background dots: 2 000-row random sample from cleaned dataset (grey, opacity 0.3)
- Coloured dots: 8 recommended tracks (colour by quadrant)
- Star marker: user's target point
- Quadrant dividers drawn as dashed lines at x=0.5, y=0.5

---

### `ui/app.py`

```
Purpose : Streamlit entry point
Run     : streamlit run ui/app.py
```

**Pages (Streamlit tabs or sidebar radio):**

| Tab / Page | Description |
|------------|-------------|
| **Survey** | Multi-step survey → recommendation results |
| **Data Lab** | Data cleaning pipeline demo for professor |
| **About** | Project info, tech stack, dataset credit |

---

## 7. UI Specification

### Global styling (injected via `st.markdown`)

```css
/* Dark theme with gradient sidebar */
background-color: #0f0f1a;
accent:           #a855f7  (purple);
card-bg:          #1a1a2e;
card-border:      1px solid #a855f7;
font:             'Inter', sans-serif;
```

### Survey flow

```
Step 1 → Step 2 → Step 3 → Step 4 → [Computing... spinner] → Results
```

- One question per screen, `st.session_state["step"]` tracks position.
- Four styled option buttons (not radio), clicking advances to next step.
- Progress bar at top (25 % → 50 % → 75 % → 100 %).
- Back button available on steps 2–4.

### Results screen

```
┌──────────────────────────────────────────────────────┐
│  Mood: 😄 Joyful          [Genre filter multiselect] │
├──────────────────────────────────────────────────────┤
│ ┌────────────┐ ┌────────────┐ ┌────────────┐ ...    │
│ │ Track Name │ │ Track Name │ │ Track Name │        │
│ │ Artist     │ │ Artist     │ │ Artist     │        │
│ │ Album      │ │ Album      │ │ Album      │        │
│ │ ⚡ 0.87    │ │ ⚡ 0.82    │ │ ⚡ 0.79    │        │
│ │ 💜 0.91    │ │ 💜 0.88    │ │ 💜 0.84    │        │
│ │ 🎵 142 BPM │ │ 🎵 138 BPM │ │ 🎵 135 BPM │        │
│ │ 94% match  │ │ 91% match  │ │ 89% match  │        │
│ └────────────┘ └────────────┘ └────────────┘        │
│                                                      │
│ [Visualisations: Chart 1 | Chart 2 | Chart 3]       │
└──────────────────────────────────────────────────────┘
```

- 2-column responsive grid using `st.columns(2)` (wraps to 1 column on mobile).
- Loading spinner shown during `recommender.recommend()` call.
- "Retake Survey" button resets `st.session_state`.

---

## 8. Data Lab — Professor Demo Mode

**Purpose:** Show the professor exactly how the cleaning pipeline works, step by step.

**Location:** Second tab in the Streamlit app (`ui/app.py`).

### Layout

```
┌─────────────────────────────────────────────────────────┐
│  Data Cleaning Pipeline — Live Demo                     │
│                                                         │
│  [▶ Run Pipeline]                                        │
│                                                         │
│  Step 1: Load raw CSV                                    │
│    Before: 232,725 rows | After: 232,725 | Removed: 0   │
│    ──────────────────────────────────────────────────   │
│  Step 2: Remove duplicates                               │
│    Before: 232,725 rows | After: 230,878 | Removed: 1,847│
│    ──────────────────────────────────────────────────   │
│  Step 3: Remove null rows                                │
│    Before: 230,878 rows | After: 230,566 | Removed: 312  │
│    ──────────────────────────────────────────────────   │
│  Step 4: Clip out-of-range values                        │
│    Adjustments: energy×0, valence×2, acousticness×0,    │
│    tempo×14                                             │
│    ──────────────────────────────────────────────────   │
│  Step 5: Normalise tempo → tempo_norm                   │
│    Tempo range: 0.0 – 244.95 → normalised to [0, 1]     │
│                                                         │
│  ═══════════════════════════════════════════            │
│  BEFORE sample (5 rows)  │  AFTER sample (5 rows)       │
│  [dataframe table]       │  [dataframe table]           │
│                                                         │
│  Raw row count : 232,725                                 │
│  Clean row count: 230,566 (99.07% retained)             │
│  Genres found : 26                                      │
│  Download cleaned CSV  [⬇ button]                       │
└─────────────────────────────────────────────────────────┘
```

### Manual demo script (for presenting live)

1. Open the app → click **Data Lab** tab.
2. Click **▶ Run Pipeline** — watch each step animate in.
3. Show **Before / After** tables side by side.
4. Point to the **clip adjustment counts** (shows real dirty data was caught).
5. Show **99.07 % rows retained** — demonstrates data quality.
6. Offer to download the cleaned CSV with the **⬇** button.

---

## 9. Visualisations Summary

| # | Chart | Library | Where shown |
|---|-------|---------|-------------|
| 1 | User mood vs recommended avg (grouped bar) | Plotly | Results → Chart 1 tab |
| 2 | Feature correlation heatmap | Seaborn / Matplotlib | Results → Chart 2 tab |
| 3 | Valence vs energy scatter (coloured quadrants) | Plotly | Results → Chart 3 tab |

Charts are displayed inside a `st.tabs(["Mood Comparison", "Correlation", "Scatter"])` container below the result cards.

---

## 10. File Structure

```
music based song suggester/
├── SPEC.md
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── SpotifyFeatures.csv          ← download from Kaggle (not committed)
│   └── processed/
│       └── SpotifyFeatures_clean.csv    ← auto-generated by validator
│
├── logs/
│   └── validation_summary.txt           ← auto-generated on first run
│
├── src/
│   ├── __init__.py
│   ├── validator.py
│   ├── mood_mapper.py
│   ├── recommender.py
│   └── visualizer.py
│
├── ui/
│   ├── app.py
│   └── styles.css                       ← optional extracted CSS
│
└── tests/
    ├── __init__.py
    ├── test_validator.py
    ├── test_mood_mapper.py
    └── test_recommender.py
```

---

## 11. Tests (pytest)

### `tests/test_validator.py`
- CSV loads without error
- Duplicates are correctly removed (mock dataframe with known dupes)
- Nulls in feature columns are dropped
- Values are correctly clipped (energy > 1.0 becomes 1.0, tempo < 0 becomes 0.0)
- `tempo_norm` is in [0, 1] after normalisation
- `get_cleaning_steps_log()` returns correct step count and structure

### `tests/test_mood_mapper.py`
- All 4×4×4×4 = 256 input combinations produce a valid vector
- Quadrant labels are correct for edge inputs (e.g., q1=4, q2=4 → "Joyful")
- `vector` field has length 4
- All vector values are in [0, 1]

### `tests/test_recommender.py`
- `recommend()` returns exactly `k=8` rows given sufficient data
- `similarity_pct` is in [0, 100]
- Results are sorted descending by `similarity_pct`
- Genre filter correctly limits results to specified genres
- Empty genre filter returns results from all genres

**Coverage target:** ≥ 80 % for `src/` modules.

Run command:
```bash
pytest tests/ -v --tb=short
```

---

## 12. requirements.txt (pinned)

```
streamlit==1.33.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
plotly==5.22.0
seaborn==0.13.2
matplotlib==3.8.4
pytest==8.2.0
pytest-cov==5.0.0
```

---

## 13. Environment Setup

```bash
# 1. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place dataset
# Download SpotifyFeatures.csv from Kaggle and place at:
#   data/raw/SpotifyFeatures.csv

# 4. Run the app (auto-triggers cleaning pipeline on first launch)
streamlit run ui/app.py
```

---

## 14. 3-Day Build Timeline

### Day 1 — Data foundation
- [ ] Download and inspect `SpotifyFeatures.csv`
- [ ] Implement `src/validator.py` (full cleaning pipeline + step log)
- [ ] Write `tests/test_validator.py` (TDD: write tests first)
- [ ] Verify cleaned CSV and `validation_summary.txt` output

### Day 2 — Core ML + charts
- [ ] Implement `src/mood_mapper.py` + `tests/test_mood_mapper.py`
- [ ] Implement `src/recommender.py` + `tests/test_recommender.py`
- [ ] Implement `src/visualizer.py` (all 3 charts)
- [ ] Run `pytest tests/ -v` and hit ≥ 80 % coverage

### Day 3 — UI + polish
- [ ] Build `ui/app.py` — survey flow, results page, genre filter
- [ ] Build Data Lab tab with step-by-step animation
- [ ] Inject dark theme CSS, style result cards
- [ ] End-to-end walkthrough: fill survey → view results → demo Data Lab
- [ ] Write `README.md` with setup instructions and screenshots

---

## 15. Acceptance Criteria

| # | Criterion | How to verify |
|---|-----------|---------------|
| AC-1 | Validator produces clean CSV on first run | Check `data/processed/` and `logs/` |
| AC-2 | Survey completes in ≤ 4 steps with progress bar | Manual walkthrough |
| AC-3 | Exactly 8 recommendations returned | Check results screen |
| AC-4 | Each card shows track, artist, album, features, % match | Visual check |
| AC-5 | Genre filter narrows results correctly | Select one genre, verify all 8 match |
| AC-6 | All 3 charts render without error | Click each chart tab |
| AC-7 | Data Lab shows all 5 cleaning steps + before/after | Demo tab walkthrough |
| AC-8 | pytest passes with ≥ 80 % coverage on `src/` | `pytest --cov=src tests/` |
| AC-9 | No API keys or internet connection required to run | Run on offline machine |
| AC-10 | `requirements.txt` is complete and pinned | Fresh venv install test |
