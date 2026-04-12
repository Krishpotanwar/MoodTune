"""
app.py — MoodTune Streamlit entry point.

Run from the project root:
    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make src/ importable regardless of working directory
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.mood_mapper import SURVEY_QUESTIONS, map_to_vector
from src.recommender import build_model, recommend
from src.validator import get_cleaning_steps_log, run_pipeline
from src.visualizer import (
    feature_correlation_heatmap,
    mood_cluster_scatter,
    mood_vs_recommended_bar,
)

# ── page config (must be first Streamlit call) ─────────────────────────────────

st.set_page_config(
    page_title="MoodTune",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── inject CSS ─────────────────────────────────────────────────────────────────

_CSS_PATH = Path(__file__).parent / "styles.css"
if _CSS_PATH.exists():
    # Use st.markdown with unsafe_allow_html for CSS injection.
    # st.html() is not available on all Streamlit Cloud deployments.
    st.markdown(
        f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )

# ── paths ──────────────────────────────────────────────────────────────────────

_RAW_CSV    = _PROJECT_ROOT / "data" / "raw"       / "spotify_tracks.csv"
_CLEAN_CSV  = _PROJECT_ROOT / "data" / "processed" / "spotify_tracks_clean.csv"
_SAMPLE_CSV = _PROJECT_ROOT / "data" / "sample"    / "spotify_sample.csv"
_LOG_PATH   = _PROJECT_ROOT / "logs"               / "validation_summary.txt"

# True when running on cloud without the full Kaggle dataset
_DEMO_MODE: bool = not _RAW_CSV.exists() and not _CLEAN_CSV.exists()

# ── cached data helpers ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_clean_df() -> pd.DataFrame:
    """
    Load the dataset.

    Priority:
    1. Pre-cleaned CSV (data/processed/) — fastest, used after first local run.
    2. Raw CSV (data/raw/) — triggers the cleaning pipeline on first local run.
    3. Bundled sample (data/sample/) — fallback for cloud/demo deployments.
    """
    if _CLEAN_CSV.exists():
        return pd.read_csv(_CLEAN_CSV)
    if _RAW_CSV.exists():
        run_pipeline(_RAW_CSV, _CLEAN_CSV, _LOG_PATH)
        return pd.read_csv(_CLEAN_CSV)
    # Cloud/demo fallback — small pre-cleaned sample shipped with the repo
    return pd.read_csv(_SAMPLE_CSV)


@st.cache_resource(show_spinner=False)
def _get_model(_df_hash: int):
    """Fit and cache the NearestNeighbors model (keyed by dataframe hash)."""
    df = _load_clean_df()
    return build_model(df)


# ── session state initialisation ──────────────────────────────────────────────

def _init_state() -> None:
    defaults: dict = {
        "step":         0,
        "answers":      {},
        "user_vector":  None,
        "results":      None,
        "genre_filter": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# ── component: progress bar ────────────────────────────────────────────────────

def _progress_bar(current: int, total: int = 4) -> None:
    pct = int(current / total * 100)
    st.markdown(
        f"""
        <div class="progress-wrapper">
          <div class="progress-bar" style="width:{pct}%"></div>
          <span class="progress-label">Step {current} of {total}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── component: song card HTML ──────────────────────────────────────────────────

def _song_card_html(row: pd.Series) -> str:
    album = str(row.get("album", ""))
    album_line = f'<p class="card-album">{album}</p>' if album and album != "nan" else ""
    return f"""
    <div class="song-card">
      <div class="card-match">{row['similarity_pct']:.0f}% match</div>
      <div class="card-track">{row['track_name']}</div>
      <div class="card-artist">{row['artist_name']}</div>
      {album_line}
      <div class="card-features">
        ⚡&nbsp;{row['energy']:.2f}&emsp;
        💜&nbsp;{row['valence']:.2f}&emsp;
        🎵&nbsp;{row['tempo_norm']:.2f}
      </div>
    </div>
    """

# ══════════════════════════════════════════════════════════════════════════════
# Survey tab
# ══════════════════════════════════════════════════════════════════════════════

def _render_survey_tab() -> None:
    step = st.session_state["step"]

    # ── Step 0: Welcome ────────────────────────────────────────────────────────
    if step == 0:
        st.markdown(
            """
            <div class="hero">
              <h1>🎵 MoodTune</h1>
              <p class="hero-subtitle">
                Tell us how you feel — we'll find your perfect soundtrack.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col = st.columns([1, 2, 1])[1]
        with col:
            if st.button("▶  Start Survey", use_container_width=True, type="primary"):
                st.session_state["step"] = 1
                st.rerun()
        return

    # ── Steps 1–4: One question per step ──────────────────────────────────────
    if 1 <= step <= 4:
        _progress_bar(step)
        q = SURVEY_QUESTIONS[step - 1]
        st.markdown(f"### {q['question']}")

        option_labels = [opt["label"] for opt in q["options"]]

        # Restore previously-selected option if user navigated back
        prev_answer = st.session_state["answers"].get(f"q{step}")
        default_idx = (prev_answer - 1) if prev_answer else None

        selected = st.radio(
            label="Choose one:",
            options=option_labels,
            index=default_idx,
            key=f"q{step}_radio",
            label_visibility="collapsed",
        )

        col_back, _, col_next = st.columns([1, 2, 1])
        with col_back:
            if step > 1 and st.button("← Back", use_container_width=True):
                st.session_state["step"] -= 1
                st.rerun()
        with col_next:
            disabled = selected is None
            if st.button("Next →", use_container_width=True, type="primary", disabled=disabled):
                st.session_state["answers"][f"q{step}"] = option_labels.index(selected) + 1
                st.session_state["step"] += 1
                # Clear cached results when answers change
                st.session_state["user_vector"] = None
                st.session_state["results"]     = None
                st.rerun()
        return

    # ── Step 5: Compute and show results ──────────────────────────────────────
    if step == 5:
        _progress_bar(4, 4)
        answers = st.session_state["answers"]

        if st.session_state["user_vector"] is None:
            with st.spinner("🎧 Analysing your mood…"):
                vec = map_to_vector(
                    answers["q1"], answers["q2"],
                    answers["q3"], answers["q4"],
                )
                st.session_state["user_vector"] = vec

        if st.session_state["results"] is None:
            with st.spinner("Finding your soundtrack…"):
                df    = _load_clean_df()
                model = _get_model(id(df))
                results = recommend(
                    model, df,
                    st.session_state["user_vector"]["vector"],
                    genre_filter=st.session_state["genre_filter"],
                )
                st.session_state["results"] = results

        _render_results()


def _render_results() -> None:
    user_vector = st.session_state["user_vector"]
    results     = st.session_state["results"]
    df          = _load_clean_df()

    # ── Mood badge + retake button ─────────────────────────────────────────────
    col_badge, col_retake = st.columns([3, 1])
    with col_badge:
        st.markdown(
            f'<div class="mood-badge">'
            f'{user_vector["mood_emoji"]}&nbsp; Mood detected: '
            f'<b>{user_vector["mood_label"]}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_retake:
        if st.button("↺  Retake Survey", use_container_width=True):
            st.session_state.update({
                "step": 0, "answers": {},
                "user_vector": None, "results": None, "genre_filter": [],
            })
            st.rerun()

    # ── Genre filter ──────────────────────────────────────────────────────────
    if "genre" in df.columns:
        genres = sorted(df["genre"].dropna().unique().tolist())
        selected_genres = st.multiselect(
            "Filter by genre (optional — leave empty for all genres):",
            options=genres,
            default=st.session_state.get("genre_filter", []),
            placeholder="All genres",
        )
        if selected_genres != st.session_state["genre_filter"]:
            st.session_state["genre_filter"] = selected_genres
            model = _get_model(id(df))
            results = recommend(
                model, df,
                user_vector["vector"],
                genre_filter=selected_genres,
            )
            st.session_state["results"] = results

    if results is None or results.empty:
        st.warning("No tracks found for this genre selection. Try removing the filter.")
        return

    # ── Result cards (2-column grid) ──────────────────────────────────────────
    st.markdown("### 🎵 Your Soundtrack")
    left, right = st.columns(2)
    for i, (_, row) in enumerate(results.iterrows()):
        col = left if i % 2 == 0 else right
        with col:
            st.markdown(_song_card_html(row), unsafe_allow_html=True)

    # ── Visualisations ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Visualisations")
    tab_bar, tab_heat, tab_scatter = st.tabs([
        "📊 Mood Comparison",
        "🔥 Feature Correlation",
        "🌐 Quadrant Map",
    ])
    with tab_bar:
        st.plotly_chart(
            mood_vs_recommended_bar(user_vector, results),
            use_container_width=True,
        )
    with tab_heat:
        st.pyplot(feature_correlation_heatmap(df))
    with tab_scatter:
        st.plotly_chart(
            mood_cluster_scatter(df, user_vector, results),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Data Lab tab  (professor demo)
# ══════════════════════════════════════════════════════════════════════════════

def _render_data_lab_tab() -> None:
    st.markdown("## 🔬 Data Cleaning Pipeline — Live Demo")
    st.markdown(
        "Click **▶ Run Pipeline** to watch each cleaning step execute in real time "
        "and see exactly how raw data is transformed before training."
    )

    if not _RAW_CSV.exists():
        st.error(
            f"`{_RAW_CSV}` not found.\n\n"
            "**Download the dataset from Kaggle**, rename the file to "
            "`spotify_tracks.csv`, and place it at `data/raw/spotify_tracks.csv`:\n\n"
            "https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset"
        )
        return

    if st.button("▶  Run Pipeline", type="primary"):
        raw_df: pd.DataFrame | None = None
        clean_df: pd.DataFrame | None = None

        placeholder = st.empty()
        with placeholder.container():
            with st.spinner("Cleaning data…"):
                raw_df   = pd.read_csv(_RAW_CSV)
                clean_df = run_pipeline(_RAW_CSV, _CLEAN_CSV, _LOG_PATH)
                steps    = get_cleaning_steps_log()

        placeholder.empty()
        st.success(f"Pipeline complete — {len(clean_df):,} clean rows ready.")

        # ── Step-by-step breakdown ─────────────────────────────────────────────
        st.markdown("### Cleaning Steps")
        for s in steps:
            removed_badge = f"  ·  **{s['removed']:,} removed**" if s["removed"] > 0 else ""
            with st.expander(f"Step {s['step']}: {s['name']}{removed_badge}", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows before", f"{s['before']:,}")
                c2.metric("Rows after",  f"{s['after']:,}")
                c3.metric("Removed",     f"{s['removed']:,}",
                          delta=f"-{s['removed']:,}" if s["removed"] > 0 else None,
                          delta_color="inverse")
                if s["detail"]:
                    st.caption(f"ℹ️  {s['detail']}")

        # ── Before / After comparison ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Before vs After (first 5 rows)")
        c_raw, c_clean = st.columns(2)
        with c_raw:
            st.markdown("**Raw data**")
            st.dataframe(raw_df.head(5), use_container_width=True)
        with c_clean:
            st.markdown("**Cleaned data**")
            st.dataframe(clean_df.head(5), use_container_width=True)

        # ── Summary metrics ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Summary")
        raw_n   = len(raw_df)
        clean_n = len(clean_df)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Raw rows",    f"{raw_n:,}")
        s2.metric("Clean rows",  f"{clean_n:,}")
        s3.metric("Retained",    f"{clean_n / raw_n * 100:.1f}%")
        s4.metric("Genres",      str(clean_df["genre"].nunique()))

        # ── Download button ────────────────────────────────────────────────────
        st.download_button(
            label="⬇  Download cleaned CSV",
            data=clean_df.to_csv(index=False).encode("utf-8"),
            file_name="spotify_tracks_clean.csv",
            mime="text/csv",
        )

        # ── Validation log ─────────────────────────────────────────────────────
        if _LOG_PATH.exists():
            with st.expander("📄 Full validation log"):
                st.code(_LOG_PATH.read_text(encoding="utf-8"), language="text")

        # Invalidate cached df so results page uses fresh data
        _load_clean_df.clear()


# ══════════════════════════════════════════════════════════════════════════════
# About tab
# ══════════════════════════════════════════════════════════════════════════════

def _render_about_tab() -> None:
    st.markdown("""
## 🎵 MoodTune — Mood-Based Music Recommender

A college project demonstrating a complete, offline machine-learning pipeline:
**raw CSV → validation → cosine-similarity recommendation → interactive UI**.

---

### Architecture

```
data/raw/spotify_tracks.csv
        │
        ▼
src/validator.py ─────────────────► data/processed/spotify_tracks_clean.csv
                                              │
src/mood_mapper.py ──► target vector          │
 [energy, valence,                            ▼
  tempo_norm,        src/recommender.py (NearestNeighbors cosine)
  acousticness]              │
                             ▼
                    Top-8 matching tracks
                             │
                    src/visualizer.py ──► 3 charts
                             │
                    ui/app.py (Streamlit)  ◄── you are here
```

---

### Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit 1.33 |
| ML | scikit-learn NearestNeighbors (cosine) |
| Data | pandas 2.2 · numpy 1.26 |
| Charts | Plotly 5.22 · Seaborn 0.13 |
| Dataset | spotify_tracks.csv (Kaggle · ~114k tracks · CC0) |

---

### Dataset Credit

[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
by **maharshipandya** on Kaggle (CC0 Public Domain license).

---

### Known Limitations

- No audio playback (offline, no API keys needed)
- Recommendations are purely feature-based — no collaborative filtering
- Dataset is static (2020 snapshot, no live Spotify data)
- Genre labels from the dataset may not match user expectations exactly
""")


# ══════════════════════════════════════════════════════════════════════════════
# Main layout
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="app-header"><h2>🎵 MoodTune</h2></div>', unsafe_allow_html=True)

# Check if dataset is available before trying to load it
if not _RAW_CSV.exists() and not _CLEAN_CSV.exists():
    st.warning(
        "⚠️  Dataset not found. Open the **Data Lab** tab for download instructions, "
        "or place `spotify_tracks.csv` at `data/raw/spotify_tracks.csv` and restart."
    )

if _DEMO_MODE:
    st.info(
        "🎭 **Demo mode** — running on a 200-track sample dataset. "
        "Download the full 114k-track dataset from "
        "[Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) "
        "to run locally with the complete recommender.",
        icon="ℹ️",
    )

tab_survey, tab_lab, tab_about = st.tabs(["🎵 Survey", "🔬 Data Lab", "ℹ️ About"])

with tab_survey:
    _render_survey_tab()

with tab_lab:
    _render_data_lab_tab()

with tab_about:
    _render_about_tab()
