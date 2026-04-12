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

# ══════════════════════════════════════════════════════════════════════════════
# Survey tab
# ══════════════════════════════════════════════════════════════════════════════

def _render_survey_tab() -> None:
    step = st.session_state["step"]

    # ── Step 0: Welcome ────────────────────────────────────────────────────────
    if step == 0:
        st.title("🎵 MoodTune")
        st.caption("Tell us how you feel — we'll find your perfect soundtrack.")
        st.divider()
        col = st.columns([1, 2, 1])[1]
        with col:
            if st.button("▶️  Start Survey", use_container_width=True, type="primary"):
                st.session_state["step"] = 1
                st.rerun()
        return

    # ── Steps 1–4: One question per step ──────────────────────────────────────
    if 1 <= step <= 4:
        pct = int(step / 4 * 100)
        st.progress(pct / 100, text=f"Step {step} of 4")
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
        st.success(
            f"{user_vector['mood_emoji']} **Mood: {user_vector['mood_label']}**"
        )
    with col_retake:
        if st.button("↺ Retake Survey", use_container_width=True):
            st.session_state.update({
                "step": 0, "answers": {},
                "user_vector": None, "results": None, "genre_filter": [],
            })
            st.rerun()

    # ── Genre filter ──────────────────────────────────────────────────────────
    if "genre" in df.columns:
        genres = sorted(df["genre"].dropna().unique().tolist())
        selected_genres = st.multiselect(
            "Filter by genre (optional):",
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

    # ── Result cards (2-column grid using native Streamlit containers) ─────────
    st.markdown("### 🎵 Your Soundtrack")
    left, right = st.columns(2)
    for i, (_, row) in enumerate(results.iterrows()):
        col = left if i % 2 == 0 else right
        with col:
            with st.container(border=True):
                st.caption(f"**{row['similarity_pct']:.0f}% match**")
                st.markdown(f"**{row['track_name']}**")
                st.caption(row['artist_name'])
                album = str(row.get("album", ""))
                if album and album != "nan":
                    st.caption(f"_{album}_")
                st.caption(
                    f"⚡ Energy: {row['energy']:.2f}  "
                    f"|  💜 Valence: {row['valence']:.2f}  "
                    f"|  🎵 Tempo: {row['tempo_norm']:.2f}"
                )

    # ── Visualisations ────────────────────────────────────────────────────────
    st.divider()
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
    st.caption(
        "Click **▶ Run Pipeline** to watch each cleaning step execute in real time."
    )

    if not _RAW_CSV.exists():
        if _DEMO_MODE:
            st.info(
                "Running in **demo mode** with a 200-track sample. "
                "The Data Lab pipeline demo requires the full dataset. "
                "Download it from "
                "[Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) "
                "and place it at `data/raw/spotify_tracks.csv`."
            )
        else:
            st.error(
                f"`{_RAW_CSV}` not found.\n\n"
                "Download the dataset from Kaggle and place it at "
                "`data/raw/spotify_tracks.csv`:\n\n"
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
        st.divider()
        st.markdown("### Before vs After (first 5 rows)")
        c_raw, c_clean = st.columns(2)
        with c_raw:
            st.markdown("**Raw data**")
            st.dataframe(raw_df.head(5), use_container_width=True)
        with c_clean:
            st.markdown("**Cleaned data**")
            st.dataframe(clean_df.head(5), use_container_width=True)

        # ── Summary metrics ────────────────────────────────────────────────────
        st.divider()
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
| UI | Streamlit 1.33+ |
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

# Demo mode banner — only shown when full dataset is absent
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
