"""Tests for src/data_loader.py."""
from __future__ import annotations

import hashlib
from pathlib import Path

from src.data_loader import _sha256_file


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("moodtune-checksum", encoding="utf-8")

    digest = _sha256_file(sample)
    expected = hashlib.sha256(b"moodtune-checksum").hexdigest()

    assert digest == expected
