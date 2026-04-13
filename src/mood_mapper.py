"""
mood_mapper.py — Survey question definitions and answer-to-vector mapping.

Maps the user's 4 survey answers (each 1–4) to a normalised feature vector
[energy, valence, tempo_norm, acousticness] used by the recommender, and
derives the mood quadrant label (Joyful / Angry / Relaxed / Depressed).
"""

from __future__ import annotations

# ── survey definitions ─────────────────────────────────────────────────────────

SURVEY_QUESTIONS: list[dict] = [
    {
        "id":       "q1",
        "question": "How energetic do you feel right now?",
        "feature":  "energy",
        "options": [
            {"label": "Exhausted", "score": 0.10},
            {"label": "Low",       "score": 0.35},
            {"label": "Moderate",  "score": 0.65},
            {"label": "Pumped",    "score": 0.90},
        ],
    },
    {
        "id":       "q2",
        "question": "How positive is your mood?",
        "feature":  "valence",
        "options": [
            {"label": "Very negative", "score": 0.10},
            {"label": "Low",           "score": 0.35},
            {"label": "Neutral",       "score": 0.60},
            {"label": "Great",         "score": 0.90},
        ],
    },
    {
        "id":       "q3",
        "question": "What are you currently doing?",
        "feature":  "tempo_norm",
        "options": [
            {"label": "Resting",           "score": 0.15},
            {"label": "Working / Studying", "score": 0.40},
            {"label": "Working out",        "score": 0.70},
            {"label": "Partying",           "score": 0.90},
        ],
    },
    {
        "id":       "q4",
        "question": "Are you alone or with others?",
        "feature":  "acousticness",
        "options": [
            {"label": "Alone",            "score": 0.85},
            {"label": "With 1–2 friends", "score": 0.55},
            {"label": "Small group",      "score": 0.30},
            {"label": "Big crowd",        "score": 0.10},
        ],
    },
]

# ── mood quadrant lookup ───────────────────────────────────────────────────────
# Determined from energy (Q1) and valence (Q2) scores.
# Threshold is 0.5 on both axes.

_QUADRANT_TABLE: dict[tuple[bool, bool], tuple[str, str]] = {
    (True,  True):  ("Joyful",    "😄"),   # high energy, high valence
    (True,  False): ("Angry",     "😤"),   # high energy, low  valence
    (False, True):  ("Relaxed",   "😌"),   # low  energy, high valence
    (False, False): ("Depressed", "😔"),   # low  energy, low  valence
}

_QUADRANT_THRESHOLD: float = 0.5


# ── public API ─────────────────────────────────────────────────────────────────

def map_to_vector(q1: int, q2: int, q3: int, q4: int) -> dict:
    """
    Convert four survey answers (each 1–4) to a normalised feature vector.

    Args:
        q1: Energy answer.      1 = Exhausted … 4 = Pumped.
        q2: Valence answer.     1 = Very negative … 4 = Great.
        q3: Tempo answer.       1 = Resting … 4 = Partying.
        q4: Acousticness answer. 1 = Alone … 4 = Big crowd.

    Returns:
        dict with keys:
            energy       (float)
            valence      (float)
            tempo_norm   (float)
            acousticness (float)
            vector       (list[float])  — ordered [energy, valence, tempo_norm, acousticness]
            coordinate   (tuple[float, float]) — ordered (valence, energy)
            mood_label   (str)          — one of Joyful / Angry / Relaxed / Depressed
            mood_emoji   (str)

    Raises:
        ValueError: if any answer is outside 1–4.
    """
    inputs = {"q1": q1, "q2": q2, "q3": q3, "q4": q4}
    for name, val in inputs.items():
        if val not in (1, 2, 3, 4):
            raise ValueError(
                f"{name} must be an integer in 1–4, got {val!r}"
            )

    scores: list[float] = [
        SURVEY_QUESTIONS[i]["options"][answers - 1]["score"]
        for i, answers in enumerate([q1, q2, q3, q4])
    ]
    energy, valence, tempo_norm, acousticness = scores

    high_energy  = energy  >= _QUADRANT_THRESHOLD
    high_valence = valence >= _QUADRANT_THRESHOLD
    mood_label, mood_emoji = _QUADRANT_TABLE[(high_energy, high_valence)]

    return {
        "energy":       energy,
        "valence":      valence,
        "tempo_norm":   tempo_norm,
        "acousticness": acousticness,
        "vector":       scores,
        "coordinate":   (valence, energy),
        "mood_label":   mood_label,
        "mood_emoji":   mood_emoji,
    }


# ── standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    examples = [
        ((4, 4, 4, 4), "Pumped + Great + Partying + Big crowd → Joyful"),
        ((4, 1, 4, 4), "Pumped + Very negative + Partying + Big crowd → Angry"),
        ((1, 4, 1, 1), "Exhausted + Great + Resting + Alone → Relaxed"),
        ((1, 1, 1, 1), "Exhausted + Very negative + Resting + Alone → Depressed"),
    ]
    print("MoodTune — Mood Mapper Verification")
    print("=" * 50)
    for (q1, q2, q3, q4), description in examples:
        result = map_to_vector(q1, q2, q3, q4)
        print(f"\n  {description}")
        print(f"  Vector : {result['vector']}")
        print(f"  Quadrant: {result['mood_emoji']} {result['mood_label']}")
