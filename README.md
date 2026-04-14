# 🎵 MoodTune — Navigate Music Through Emotional Space

> **MoodTune** transforms 81,000+ songs into an interactive 3D mood map, letting you journey from how you feel now to how you want to feel — with AI-curated playlists that bridge the emotional gap.

**🌐 [Live Demo](https://moodtune-cayerjwf67tt46ppthmeei.streamlit.app)**

---

## What It Does

Most music recommenders match **one static mood**. MoodTune treats music discovery like **path planning through emotional space** — you set a start mood and a target mood, and the app generates an 18-step playlist that transitions smoothly between them, visualized as a visible path through a scatter plot of real songs.

---

## Three Ways to Set Your Mood

### 1. 🎯 Mood Space (Interactive Scatter)
Click directly on a 5,000-song scatter plot. Each point is a real track — click one to set your start or target mood. The chart shows four emotional quadrants:
- **Joyful** 😄 — high energy, positive
- **Angry** 😤 — high energy, negative
- **Relaxed** 😌 — low energy, positive
- **Depressed** 😔 — low energy, negative

### 2. ✍️ Natural Language Text
Describe your mood in plain English: *"I just finished a long exam and want to zone out"* → MoodTune maps it to coordinates using keyword matching with confidence scoring.

### 3. 📋 Classic Survey
A guided 4-question fallback that maps your answers to `(valence, energy)` coordinates.

---

## The Journey Algorithm

Once you've set start and target moods:

1. **KDTree Indexing** — The full 81k-song dataset is indexed using SciPy's `KDTree` for fast nearest-neighbour queries in mood space.
2. **Waypoint Interpolation** — The app calculates intermediate mood points between your start and target, creating a smooth emotional arc.
3. **Song Selection** — For each waypoint, it finds nearby tracks while penalizing abrupt tempo jumps (`transition_score = mood_distance + 0.30 × |tempo_diff|`).
4. **Visualization** — The final playlist is drawn as a connected path through mood space so you can see exactly how the emotional transition unfolds.

---

## 3D Mood Space

All 81,000+ tracks plotted across three axes:
- **X: Valence** (sad ← → happy)
- **Y: Energy** (chill ← → intense)
- **Z: Danceability** (still ← → groove)

Drag to rotate, scroll to zoom. Tracks are colour-coded by genre using the Plasma colorscale (dark purple → hot pink → yellow). The 3D view reveals structure invisible in 2D — which genres cluster where in emotional space, and how danceability adds a third dimension to mood.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit + Plotly (WebGL) |
| **Data Processing** | Pandas + NumPy + custom validator pipeline |
| **ML/Algorithms** | SciPy KDTree, scikit-learn nearest-neighbour, NLP mood mapping |
| **Dataset** | 81k Spotify tracks (14 audio features) via HuggingFace Hub |
| **Visualization** | Plotly 2D scatter (5k sample) + 3D scatter (25k sample) |
| **Styling** | Custom CSS — baby pink dark theme with liquid glassmorphism |
| **Deployment** | Streamlit Cloud (free tier) |

---

## Local Setup

```bash
# 1. Clone and install dependencies
git clone https://github.com/Krishpotanwar/Hisab.git
cd "music based song suggester"
pip install -r requirements.txt

# 2. Run the app
streamlit run ui/app.py
```

The app auto-detects the local clean dataset at `data/processed/spotify_tracks_clean.csv` (~16MB, 81k tracks). If unavailable, it downloads from [HuggingFace Hub](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) on first run (cached for subsequent visits).

---

## Data Pipeline

The raw Kaggle CSV (`data/raw/spotify_tracks.csv`, 20MB) goes through an 8-step cleaning pipeline:
1. Drop duplicates
2. Remove rows missing audio features
3. Normalize tempo to [0, 1]
4. Standardize genre labels
5. Filter outliers (BPM > 250)
6. Validate required columns
7. Cast types
8. Export clean CSV

The pipeline runs once and caches the result. Data Lab tab shows before/after metrics and feature correlation heatmap.

---

## Project Structure

```
music based song suggester/
├── src/
│   ├── config.py           # Paths and constants
│   ├── data_loader.py      # Multi-source dataset loader (local → HuggingFace → sample)
│   ├── journey.py          # KDTree-based mood journey generation
│   ├── mood_mapper.py      # Survey-to-coordinate mapping
│   ├── nlp_mood.py         # Natural language mood parsing
│   ├── recommender.py      # Nearest-neighbour track recommendation
│   ├── validator.py        # Raw → clean data pipeline
│   └── visualizer.py       # Plotly/Matplotlib charts (2D, 3D, heatmap, progress)
├── ui/
│   ├── app.py              # Streamlit entry point (4 tabs)
│   └── styles.css          # Baby pink dark glassmorphism theme
├── data/
│   ├── raw/                # Original Kaggle CSV (20MB)
│   ├── processed/          # Clean CSV (16MB, 81k rows)
│   └── sample/             # Fallback sample (200 tracks)
├── tests/                  # Unit tests
└── requirements.txt
```

---

## Verification Checklist

- [x] Stats banner shows "81,000+ tracks" (not 200)
- [x] 2D scatter shows 5,000 sample points, click sets start/target
- [x] 3D scatter renders (rotate, zoom work), ~25k points, black bg
- [x] Journey generates 18 tracks, displays as card grid
- [x] NLP text input maps to coordinates correctly
- [x] No import errors, no missing data warnings

---

## License

MIT

---

**Built for demo purposes** — showcases KDTree path planning, NLP mood parsing, and interactive 3D visualization in a consumer-facing music app.
