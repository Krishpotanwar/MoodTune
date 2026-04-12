"""
visualizer.py — Chart generation functions for MoodTune.

All functions return figure objects (never call .show()) so Streamlit
can render them with st.plotly_chart / st.pyplot.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")   # non-interactive backend required for Streamlit
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

# ── theme constants ────────────────────────────────────────────────────────────

_BG_MAIN  = "#0f0f1a"
_BG_CARD  = "#1a1a2e"
_ACCENT   = "#a855f7"
_CYAN     = "#06b6d4"

_FEATURE_LABELS: list[str] = ["Energy", "Valence", "Tempo", "Acousticness"]
_FEATURE_KEYS:   list[str] = ["energy", "valence", "tempo_norm", "acousticness"]

_QUADRANT_COLORS: dict[str, str] = {
    "Joyful":    "#f59e0b",
    "Angry":     "#ef4444",
    "Relaxed":   "#10b981",
    "Depressed": "#6366f1",
}

_QUADRANT_LABELS: list[tuple[float, float, str]] = [
    (0.75, 0.75, "Joyful 😄"),
    (0.25, 0.75, "Angry 😤"),
    (0.75, 0.25, "Relaxed 😌"),
    (0.25, 0.25, "Depressed 😔"),
]


# ── helpers ────────────────────────────────────────────────────────────────────

def _quadrant_label(energy: float, valence: float) -> str:
    """Return the mood quadrant name for a given (energy, valence) pair."""
    if energy >= 0.5 and valence >= 0.5:
        return "Joyful"
    if energy >= 0.5 and valence < 0.5:
        return "Angry"
    if energy < 0.5 and valence >= 0.5:
        return "Relaxed"
    return "Depressed"


# ── Chart 1 ────────────────────────────────────────────────────────────────────

def mood_vs_recommended_bar(
    user_vector: dict,
    recommendations: pd.DataFrame,
) -> go.Figure:
    """
    Grouped bar chart: user target features vs average of recommended tracks.

    Args:
        user_vector:     dict with keys energy, valence, tempo_norm, acousticness.
        recommendations: DataFrame from recommender.recommend().

    Returns:
        Plotly Figure with two bar groups ("Your Mood" and "Avg Recommended").
    """
    user_vals: list[float] = [user_vector[k] for k in _FEATURE_KEYS]
    avg_vals:  list[float] = [float(recommendations[k].mean()) for k in _FEATURE_KEYS]

    fig = go.Figure(data=[
        go.Bar(
            name="Your Mood",
            x=_FEATURE_LABELS,
            y=user_vals,
            marker_color=_ACCENT,
            text=[f"{v:.2f}" for v in user_vals],
            textposition="outside",
        ),
        go.Bar(
            name="Avg Recommended",
            x=_FEATURE_LABELS,
            y=avg_vals,
            marker_color=_CYAN,
            text=[f"{v:.2f}" for v in avg_vals],
            textposition="outside",
        ),
    ])
    fig.update_layout(
        barmode="group",
        title="Your Mood vs. Average of Recommended Tracks",
        paper_bgcolor=_BG_MAIN,
        plot_bgcolor=_BG_CARD,
        font=dict(color="white", size=13),
        legend=dict(bgcolor=_BG_CARD, bordercolor="#444"),
        yaxis=dict(range=[0, 1.15], gridcolor="#333", title="Score (0–1)"),
        xaxis=dict(title="Audio Feature"),
        margin=dict(t=60, b=40),
    )
    return fig


# ── Chart 2 ────────────────────────────────────────────────────────────────────

def feature_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Seaborn correlation heatmap for the four audio features
    (plus danceability if present in the dataset).

    Args:
        df: Cleaned DataFrame from validator.run_pipeline().

    Returns:
        Matplotlib Figure (pass to st.pyplot()).
    """
    cols = list(_FEATURE_KEYS)
    if "danceability" in df.columns:
        cols.append("danceability")

    corr = df[cols].corr()
    tick_labels = [c.replace("_norm", "").replace("_", " ").title() for c in cols]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(_BG_MAIN)
    ax.set_facecolor(_BG_MAIN)

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ax=ax,
        linewidths=0.5,
        linecolor="#333",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Audio Feature Correlations", color="white", fontsize=14, pad=12)
    ax.tick_params(colors="white", labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", color="white")
    plt.setp(ax.get_yticklabels(), rotation=0,  color="white")

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors="white")

    plt.tight_layout()
    return fig


# ── Chart 3 ────────────────────────────────────────────────────────────────────

def mood_cluster_scatter(
    df: pd.DataFrame,
    user_vector: dict,
    recommendations: pd.DataFrame,
) -> go.Figure:
    """
    Scatter plot of valence (x) vs energy (y).

    - Grey dots: 2 000-row random sample from the full cleaned dataset.
    - Coloured dots: the 8 recommended tracks, colour-coded by mood quadrant.
    - Gold star: the user's target point.
    - Dashed lines divide the four quadrants at (0.5, 0.5).

    Args:
        df:              Cleaned DataFrame.
        user_vector:     dict with energy, valence keys.
        recommendations: DataFrame from recommender.recommend().

    Returns:
        Plotly Figure.
    """
    sample = df.sample(n=min(2_000, len(df)), random_state=42)
    fig = go.Figure()

    # ── background sample ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=sample["valence"],
        y=sample["energy"],
        mode="markers",
        marker=dict(color="#555", opacity=0.25, size=4),
        name="All tracks",
        hoverinfo="skip",
    ))

    # ── quadrant dividers ──────────────────────────────────────────────────────
    fig.add_vline(x=0.5, line_dash="dash", line_color="#555", opacity=0.6)
    fig.add_hline(y=0.5, line_dash="dash", line_color="#555", opacity=0.6)

    # ── quadrant labels ────────────────────────────────────────────────────────
    for x_pos, y_pos, label in _QUADRANT_LABELS:
        fig.add_annotation(
            x=x_pos, y=y_pos, text=label,
            showarrow=False,
            font=dict(color="#777", size=11),
            xanchor="center",
        )

    # ── recommended tracks ─────────────────────────────────────────────────────
    seen_quadrants: set[str] = set()
    for _, row in recommendations.iterrows():
        quadrant = _quadrant_label(float(row["energy"]), float(row["valence"]))
        show_legend = quadrant not in seen_quadrants
        seen_quadrants.add(quadrant)

        name = str(row.get("track_name", ""))
        artist = str(row.get("artist_name", ""))
        fig.add_trace(go.Scatter(
            x=[row["valence"]],
            y=[row["energy"]],
            mode="markers+text",
            marker=dict(
                color=_QUADRANT_COLORS[quadrant],
                size=12,
                line=dict(color="white", width=1.2),
            ),
            text=[name[:22] + ("…" if len(name) > 22 else "")],
            textposition="top center",
            textfont=dict(size=8, color="white"),
            name=quadrant,
            legendgroup=quadrant,
            showlegend=show_legend,
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"Artist: {artist}<br>"
                f"Energy: {row['energy']:.2f} | Valence: {row['valence']:.2f}<br>"
                f"Quadrant: {quadrant}<extra></extra>"
            ),
        ))

    # ── user star ──────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[user_vector["valence"]],
        y=[user_vector["energy"]],
        mode="markers",
        marker=dict(symbol="star", color="#f59e0b", size=20,
                    line=dict(color="white", width=1.5)),
        name="You",
        hovertemplate=(
            "<b>Your Mood</b><br>"
            f"Energy: {user_vector['energy']:.2f}<br>"
            f"Valence: {user_vector['valence']:.2f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="Mood Quadrant Map — Where Do the Recommendations Land?",
        xaxis=dict(title="Valence (Positivity) →", range=[0, 1], gridcolor="#333"),
        yaxis=dict(title="Energy →",               range=[0, 1], gridcolor="#333"),
        paper_bgcolor=_BG_MAIN,
        plot_bgcolor=_BG_CARD,
        font=dict(color="white"),
        legend=dict(bgcolor=_BG_CARD, bordercolor="#444"),
        margin=dict(t=60, b=50),
    )
    return fig


# ── standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    clean_path = project_root / "data" / "processed" / "SpotifyFeatures_clean.csv"

    if not clean_path.exists():
        print("ERROR: Run src/validator.py first to generate the cleaned CSV.")
        sys.exit(1)

    df = pd.read_csv(clean_path)

    # Fake user vector and recommendations for testing
    fake_user = {"energy": 0.9, "valence": 0.9, "tempo_norm": 0.9, "acousticness": 0.1}
    fake_recs = df.sample(8, random_state=1).copy()
    fake_recs["similarity_pct"] = [95, 93, 91, 90, 88, 87, 85, 83]

    print("Generating Chart 1 (grouped bar)…")
    fig1 = mood_vs_recommended_bar(fake_user, fake_recs)
    fig1.show()

    print("Generating Chart 2 (correlation heatmap)…")
    fig2 = feature_correlation_heatmap(df)
    plt.show()

    print("Generating Chart 3 (mood scatter)…")
    fig3 = mood_cluster_scatter(df, fake_user, fake_recs)
    fig3.show()

    print("All charts rendered successfully.")
