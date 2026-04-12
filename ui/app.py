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
import streamlit.components.v1 as components

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

# ── inject custom CSS ──────────────────────────────────────────────────────────

_CSS_FILE = Path(__file__).parent / "styles.css"
if _CSS_FILE.exists():
    st.markdown(f"<style>{_CSS_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

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

    # ── Spotify-style playlist list ───────────────────────────────────────────
    st.markdown("### 🎵 Your Soundtrack")
    st.caption(f"Based on Kaggle Spotify Tracks Dataset · {len(df):,} tracks analysed")

    # Build the playlist HTML with inline styles (components.html renders raw without sanitization)
    playlist_html = '<div class="playlist-container">'
    # Header row
    playlist_html += '''
    <div class="playlist-header">
        <span class="ph-num">#</span>
        <span class="ph-cover"></span>
        <span class="ph-title">TITLE</span>
        <span class="ph-album">ALBUM</span>
        <span class="ph-genre">GENRE</span>
        <span class="ph-match">MATCH</span>
        <span class="ph-features">FEATURES</span>
    </div>
    '''

    for i, (_, row) in enumerate(results.iterrows()):
        # Generate unique album cover gradient from audio features
        energy = float(row["energy"])
        valence = float(row["valence"])
        tempo = float(row["tempo_norm"])
        acousticness = float(row["acousticness"])

        # Map features to hue/saturation for a unique cover per song
        hue1 = int(energy * 300 + 20)         # 20-320 range
        hue2 = int(valence * 200 + 100)       # 100-300 range
        sat = int(50 + acousticness * 40)     # 50-90%
        angle = int(tempo * 360)              # gradient angle

        album = str(row.get("album", ""))
        album_display = album if album and album != "nan" else "—"
        genre = str(row.get("genre", ""))
        track_name = str(row["track_name"])
        artist = str(row["artist_name"])
        match_pct = float(row["similarity_pct"])

        # Cover gradient with musical note overlay
        cover_style = (
            f"background: linear-gradient({angle}deg, "
            f"hsl({hue1},{sat}%,35%), "
            f"hsl({hue2},{sat}%,25%));"
        )

        # Feature mini-bars
        e_w = int(energy * 100)
        v_w = int(valence * 100)
        t_w = int(tempo * 100)

        # Spotify open link (if track_id exists)
        track_id = str(row.get("track_id", ""))
        spotify_link = ""
        if track_id and track_id != "nan":
            spotify_link = f'<a href="https://open.spotify.com/track/{track_id}" target="_blank" class="spotify-link" title="Open in Spotify">▶</a>'

        playlist_html += f'''
        <div class="playlist-row">
            <span class="pr-num">{i + 1}</span>
            <div class="pr-cover" style="{cover_style}">
                <span class="cover-icon">♫</span>
            </div>
            <div class="pr-title-block">
                <div class="pr-track">{track_name} {spotify_link}</div>
                <div class="pr-artist">{artist}</div>
            </div>
            <span class="pr-album">{album_display}</span>
            <span class="pr-genre">{genre}</span>
            <span class="pr-match">{match_pct:.0f}%</span>
            <div class="pr-features">
                <div class="feat-row"><span class="feat-label">⚡</span><div class="feat-bar"><div class="feat-fill" style="width:{e_w}%"></div></div></div>
                <div class="feat-row"><span class="feat-label">💜</span><div class="feat-bar"><div class="feat-fill feat-valence" style="width:{v_w}%"></div></div></div>
                <div class="feat-row"><span class="feat-label">🎵</span><div class="feat-bar"><div class="feat-fill feat-tempo" style="width:{t_w}%"></div></div></div>
            </div>
        </div>
        '''

    playlist_html += '</div>'
    # Calculate height: header (42px) + rows (62px each) + padding (20px)
    _playlist_height = 42 + len(results) * 62 + 20
    _render_styled_html(playlist_html, height=_playlist_height)

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

def _get_css_text() -> str:
    """Read the CSS file once (for injecting into components.html iframes)."""
    css_file = Path(__file__).parent / "styles.css"
    return css_file.read_text(encoding="utf-8") if css_file.exists() else ""


def _render_styled_html(html_body: str, height: int = 200) -> None:
    """Render HTML inside a components.html iframe with theme CSS."""
    css = _get_css_text()
    full_html = f'<style>body {{ background: transparent !important; margin: 0; padding: 0; }} {css}</style>{html_body}'
    components.html(full_html, height=height, scrolling=False)


def _render_data_lab_tab() -> None:
    st.markdown("## 🔬 Data Cleaning Pipeline — Live Demo")

    # ── Kaggle dataset info card ───────────────────────────────────────────────
    dataset_html = '''
    <div class="dataset-card">
        <div class="dataset-header">
            <span class="dataset-icon">📦</span>
            <div>
                <div class="dataset-title">Spotify Tracks Dataset</div>
                <div class="dataset-source">by maharshipandya · Kaggle · CC0 Public Domain</div>
            </div>
        </div>
        <div class="dataset-stats">
            <div class="ds-stat"><span class="ds-val">114,000+</span><span class="ds-label">Tracks</span></div>
            <div class="ds-stat"><span class="ds-val">125</span><span class="ds-label">Genres</span></div>
            <div class="ds-stat"><span class="ds-val">20+</span><span class="ds-label">Audio Features</span></div>
            <div class="ds-stat"><span class="ds-val">~20 MB</span><span class="ds-label">CSV Size</span></div>
        </div>
        <a href="https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset" target="_blank" class="dataset-link">View on Kaggle →</a>
    </div>
    '''
    _render_styled_html(dataset_html, height=230)

    # ── Pipeline flow diagram ──────────────────────────────────────────────────
    pipeline_html = '''
    <div class="pipeline-flow">
        <div class="pipe-step"><span class="pipe-num">1</span><span class="pipe-name">Load CSV</span></div>
        <span class="pipe-arrow">→</span>
        <div class="pipe-step"><span class="pipe-num">2</span><span class="pipe-name">Dedup</span></div>
        <span class="pipe-arrow">→</span>
        <div class="pipe-step"><span class="pipe-num">3</span><span class="pipe-name">Drop Nulls</span></div>
        <span class="pipe-arrow">→</span>
        <div class="pipe-step"><span class="pipe-num">4</span><span class="pipe-name">Clip Range</span></div>
        <span class="pipe-arrow">→</span>
        <div class="pipe-step"><span class="pipe-num">5</span><span class="pipe-name">Norm Tempo</span></div>
        <span class="pipe-arrow">→</span>
        <div class="pipe-step"><span class="pipe-num">6</span><span class="pipe-name">Strip WS</span></div>
        <span class="pipe-arrow">→</span>
        <div class="pipe-step pipe-done"><span class="pipe-num">✓</span><span class="pipe-name">Clean</span></div>
    </div>
    '''
    _render_styled_html(pipeline_html, height=65)

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

        # ── Visual funnel showing data reduction ───────────────────────────────
        st.markdown("### 📉 Data Funnel")
        raw_n = len(raw_df)
        funnel_html = '<div class="funnel-container">'
        for s in steps:
            pct = (s["after"] / raw_n * 100) if raw_n else 100
            bar_col = "var(--pink-400)" if s["removed"] > 0 else "var(--pink-300)"
            removed_tag = f'<span class="funnel-removed">-{s["removed"]:,}</span>' if s["removed"] > 0 else ''
            funnel_html += f'''
            <div class="funnel-row">
                <span class="funnel-label">{s["name"]}</span>
                <div class="funnel-bar-bg">
                    <div class="funnel-bar-fill" style="width:{pct:.1f}%; background:{bar_col};"></div>
                </div>
                <span class="funnel-count">{s["after"]:,}</span>
                {removed_tag}
            </div>
            '''
        funnel_html += '</div>'
        _render_styled_html(funnel_html, height=40 + len(steps) * 32)

        # ── Step-by-step breakdown ─────────────────────────────────────────────
        st.divider()
        st.markdown("### Cleaning Steps")
        for s in steps:
            removed_badge = f"  ·  **{s['removed']:,} removed**" if s["removed"] > 0 else ""
            with st.expander(f"Step {s['step']}: {s['name']}{removed_badge}", expanded=s["removed"] > 0):
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
