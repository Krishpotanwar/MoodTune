"""
nlp_mood.py -- Map free-text mood descriptions onto valence-energy space.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MOOD_LEXICON_PATH


class MoodTextResult(TypedDict):
    """Return shape for free-text mood mapping."""

    valence: float
    energy: float
    coordinate: tuple[float, float]
    matched_words: list[str]
    confidence: float


@dataclass(frozen=True)
class LexiconMatch:
    """One matched mood term with its coordinate and weighting."""

    term: str
    valence: float
    energy: float
    weight: float


_TOKEN_SANITIZER = re.compile(r"[^a-z0-9\s]+")

_DEFAULT_LEXICON: dict[str, tuple[float, float]] = {
    "happy": (0.85, 0.78),
    "excited": (0.92, 0.92),
    "energetic": (0.76, 0.94),
    "pumped": (0.80, 0.95),
    "party": (0.88, 0.92),
    "celebrate": (0.90, 0.84),
    "joy": (0.92, 0.76),
    "fun": (0.82, 0.76),
    "great": (0.80, 0.68),
    "alive": (0.82, 0.86),
    "angry": (0.14, 0.92),
    "furious": (0.08, 0.96),
    "frustrated": (0.18, 0.82),
    "aggressive": (0.16, 0.90),
    "rage": (0.08, 0.95),
    "stressed": (0.22, 0.76),
    "anxious": (0.26, 0.72),
    "tense": (0.25, 0.78),
    "calm": (0.72, 0.20),
    "peaceful": (0.76, 0.16),
    "relaxed": (0.74, 0.18),
    "chill": (0.66, 0.24),
    "unwind": (0.64, 0.18),
    "zone out": (0.50, 0.14),
    "mellow": (0.60, 0.22),
    "cozy": (0.72, 0.20),
    "soothing": (0.78, 0.14),
    "tranquil": (0.80, 0.10),
    "sad": (0.14, 0.15),
    "melancholic": (0.20, 0.18),
    "lonely": (0.12, 0.20),
    "heartbroken": (0.10, 0.16),
    "tired": (0.30, 0.10),
    "exhausted": (0.24, 0.08),
    "drained": (0.20, 0.10),
    "gloomy": (0.15, 0.12),
    "nostalgic": (0.40, 0.25),
    "bittersweet": (0.35, 0.22),
    "focused": (0.56, 0.52),
    "studying": (0.50, 0.40),
    "working": (0.50, 0.56),
    "thinking": (0.50, 0.34),
    "driving": (0.60, 0.60),
    "running": (0.70, 0.84),
    "workout": (0.76, 0.92),
    "gym": (0.74, 0.90),
    "cooking": (0.66, 0.46),
    "morning": (0.66, 0.50),
    "night": (0.42, 0.24),
    "rainy": (0.34, 0.20),
    "after exams": (0.34, 0.18),
}


def _normalise_text(text: str) -> str:
    """Lowercase and simplify whitespace for phrase matching."""
    cleaned = _TOKEN_SANITIZER.sub(" ", text.lower())
    return " ".join(cleaned.split())


def load_mood_lexicon(lexicon_path: Path | None = None) -> dict[str, tuple[float, float]]:
    """
    Load the mood lexicon JSON if present, else return the built-in fallback.

    Args:
        lexicon_path: Optional explicit path to a JSON lexicon file.

    Returns:
        Mapping of term -> (valence, energy).
    """
    source = lexicon_path or MOOD_LEXICON_PATH
    if not Path(source).exists():
        return dict(_DEFAULT_LEXICON)
    loaded = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Custom mood lexicon must be a JSON object mapping term -> [valence, energy].")

    validated: dict[str, tuple[float, float]] = {}
    for term, coords in loaded.items():
        if not isinstance(term, str):
            raise ValueError("Custom mood lexicon keys must be strings.")
        if not isinstance(coords, list) or len(coords) != 2:
            raise ValueError(
                f"Custom mood lexicon entry '{term}' must be a list of two numbers: [valence, energy]."
            )
        try:
            valence = float(coords[0])
            energy = float(coords[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Custom mood lexicon entry '{term}' contains non-numeric coordinates."
            ) from exc
        if not (0.0 <= valence <= 1.0 and 0.0 <= energy <= 1.0):
            raise ValueError(
                f"Custom mood lexicon entry '{term}' must stay within [0.0, 1.0] for both coordinates."
            )
        validated[term] = (valence, energy)

    return validated


def _term_pattern(term: str) -> str:
    """Build a regex fragment that works for single words and phrases."""
    escaped = re.escape(term)
    return rf"\b{escaped}\b"


def _find_matches(text: str, lexicon: dict[str, tuple[float, float]]) -> list[LexiconMatch]:
    """Find lexicon terms in the text and assign soft position-based weights."""
    matches: list[LexiconMatch] = []
    for term in sorted(lexicon, key=len, reverse=True):
        found = re.search(_term_pattern(term), text)
        if not found:
            continue
        valence, energy = lexicon[term]
        start_ratio = found.start() / max(len(text), 1)
        weight = max(0.6, 1.25 - start_ratio)
        matches.append(LexiconMatch(term=term, valence=valence, energy=energy, weight=weight))
    return matches


# ── HuggingFace emotion model (optional, with lexicon fallback) ───────────────

# Maps j-hartmann/emotion-english-distilroberta-base labels → (valence, energy)
_EMOTION_COORDS: dict[str, tuple[float, float]] = {
    "joy":      (0.88, 0.75),
    "surprise": (0.70, 0.80),
    "neutral":  (0.50, 0.45),
    "sadness":  (0.15, 0.18),
    "disgust":  (0.20, 0.55),
    "anger":    (0.15, 0.88),
    "fear":     (0.22, 0.70),
}

_hf_pipeline: object | None = None
_hf_available: bool | None = None  # None = not tried, True/False = tried


def _load_emotion_pipeline() -> object | None:
    """Load the HF emotion pipeline once and cache in module-level variable."""
    global _hf_pipeline, _hf_available
    if _hf_available is not None:
        return _hf_pipeline  # already attempted
    try:
        from transformers import pipeline  # type: ignore

        _hf_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
        )
        _hf_available = True
    except Exception:
        _hf_pipeline = None
        _hf_available = False
    return _hf_pipeline


def _hf_text_to_mood(text: str) -> MoodTextResult:
    """Map text to mood using the HF distilroberta emotion model.

    Raises RuntimeError if the model is unavailable or inference fails.
    Caller should catch and fall through to lexicon.
    """
    model = _load_emotion_pipeline()
    if model is None:
        raise RuntimeError("HF emotion model not available")

    results = model(text)[0]  # list of {label, score}
    valence = energy = total = 0.0
    for r in results:
        label = r["label"].lower()
        coord = _EMOTION_COORDS.get(label, (0.5, 0.5))
        valence += coord[0] * r["score"]
        energy += coord[1] * r["score"]
        total += r["score"]

    if total == 0:
        raise RuntimeError("Empty scores from HF model")

    avg_valence = round(float(np.clip(valence / total, 0.0, 1.0)), 3)
    avg_energy = round(float(np.clip(energy / total, 0.0, 1.0)), 3)
    top_emotion = max(results, key=lambda x: x["score"])["label"].lower()
    confidence = round(float(max(r["score"] for r in results)), 3)

    return {
        "valence": avg_valence,
        "energy": avg_energy,
        "coordinate": (avg_valence, avg_energy),
        "matched_words": [top_emotion],
        "confidence": confidence,
    }


# ── Main entry point ──────────────────────────────────────────────────────────


def text_to_mood_vector(text: str, lexicon_path: Path | None = None) -> MoodTextResult:
    """
    Convert a free-text mood description into a mood-space coordinate.

    Tries the HuggingFace distilroberta emotion model first.
    Falls back to lexicon-based matching if the model is unavailable.

    Args:
        text: User-entered mood description.
        lexicon_path: Optional custom lexicon JSON file.

    Returns:
        Dict with valence, energy, coordinate, matched_words, and confidence.
    """
    # Try HuggingFace model first
    try:
        return _hf_text_to_mood(text)
    except Exception:
        pass  # fall through to lexicon

    normalised_text = _normalise_text(text)
    lexicon = load_mood_lexicon(lexicon_path)
    matches = _find_matches(normalised_text, lexicon)

    if not matches:
        neutral = (0.5, 0.5)
        return {
            "valence": neutral[0],
            "energy": neutral[1],
            "coordinate": neutral,
            "matched_words": [],
            "confidence": 0.0,
        }

    weights = np.array([match.weight for match in matches], dtype=float)
    valences = np.array([match.valence for match in matches], dtype=float)
    energies = np.array([match.energy for match in matches], dtype=float)

    avg_valence = float(np.clip(np.average(valences, weights=weights), 0.0, 1.0))
    avg_energy = float(np.clip(np.average(energies, weights=weights), 0.0, 1.0))
    confidence = float(min(1.0, weights.sum() / 3.0))

    return {
        "valence": avg_valence,
        "energy": avg_energy,
        "coordinate": (avg_valence, avg_energy),
        "matched_words": [match.term for match in matches],
        "confidence": confidence,
    }


if __name__ == "__main__":
    examples = [
        "I just finished a long exam and want to zone out",
        "I need something happy and energetic for the gym",
        "It is a rainy night and I feel nostalgic",
    ]
    print("MoodTune -- NLP Mood Verification")
    print("=" * 36)
    for example in examples:
        result = text_to_mood_vector(example)
        print(f"\nText: {example}")
        print(
            f"Coordinate: ({result['valence']:.2f}, {result['energy']:.2f})"
            f" | confidence={result['confidence']:.2f}"
        )
        print(f"Matched: {result['matched_words']}")
