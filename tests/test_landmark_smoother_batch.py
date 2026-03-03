"""Tests for batched matrix-multiply similarity search in LandmarkSmoother._find_match().

Issue #95: Replace O(n) dot product loop with a single BLAS matrix multiply.
The behaviour is identical — these tests document correctness, not a new feature.

RED → GREEN workflow (this is a perf refactor, so tests may pass with old code too).
"""
import numpy as np
import pytest

from modules.face_analyser import LandmarkSmoother


# ---------------------------------------------------------------------------
# Helpers (mirrors test_ema_smoothing.py helper)
# ---------------------------------------------------------------------------

def _make_face(embedding=None):
    """Minimal InsightFace-like face object with only what _find_match() needs."""

    class _Face(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

        def __setattr__(self, name, value):
            self[name] = value

    f = _Face()
    if embedding is not None:
        raw = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(raw)
        f.normed_embedding = raw / norm if norm > 0 else raw
    else:
        f.normed_embedding = None
    # _find_match only needs normed_embedding; bbox/kps not required
    return f


def _unit(v):
    """Return a unit-normalised float32 array."""
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


# Canonical test embeddings — pairwise orthogonal (cosine = 0 between any two)
_EMB_X = [1.0, 0.0, 0.0]  # unit-x
_EMB_Y = [0.0, 1.0, 0.0]  # unit-y
_EMB_Z = [0.0, 0.0, 1.0]  # unit-z


# ---------------------------------------------------------------------------
# _find_match: empty state
# ---------------------------------------------------------------------------

class TestFindMatchEmptyState:
    def test_returns_none_when_state_is_empty(self):
        s = LandmarkSmoother()
        face = _make_face(_EMB_X)
        assert s._find_match(face) is None

    def test_returns_none_for_face_without_embedding(self):
        s = LandmarkSmoother()
        # Populate state with a real entry so we reach the embedding-check path
        face_with = _make_face(_EMB_X)
        s._state = [{'embedding': _unit(_EMB_X), 'bbox': None, 'kps': None}]
        face_without = _make_face(None)
        assert s._find_match(face_without) is None


# ---------------------------------------------------------------------------
# _find_match: threshold behaviour
# ---------------------------------------------------------------------------

class TestFindMatchThreshold:
    def test_exact_match_above_threshold(self):
        """Identical embedding → cosine = 1.0 → match returned."""
        s = LandmarkSmoother()
        entry = {'embedding': _unit(_EMB_X), 'bbox': None, 'kps': None}
        s._state = [entry]
        face = _make_face(_EMB_X)
        result = s._find_match(face)
        assert result is entry

    def test_orthogonal_embedding_below_threshold(self):
        """Cosine = 0 between orthogonal vectors < IDENTITY_THRESHOLD (0.7) → None."""
        s = LandmarkSmoother()
        s._state = [{'embedding': _unit(_EMB_X), 'bbox': None, 'kps': None}]
        face = _make_face(_EMB_Y)  # orthogonal → cosine = 0
        assert s._find_match(face) is None

    def test_below_threshold_returns_none(self):
        """Similarity just below IDENTITY_THRESHOLD → no match."""
        s = LandmarkSmoother()
        # Build an embedding whose similarity to _EMB_X is 0.5 (below 0.7 threshold)
        # cos(60°) = 0.5 → use [1, sqrt(3), 0]
        low_sim_emb = np.array([1.0, 3.0 ** 0.5, 0.0], dtype=np.float32)
        low_sim_emb /= np.linalg.norm(low_sim_emb)
        s._state = [{'embedding': _unit(_EMB_X), 'bbox': None, 'kps': None}]
        face = _make_face(low_sim_emb)
        assert s._find_match(face) is None

    def test_at_threshold_returns_match(self):
        """Similarity at or just above IDENTITY_THRESHOLD (0.7) → match returned.

        Uses float64 arithmetic to construct an embedding whose cosine to _EMB_X
        is >= 0.7 after float32 rounding, avoiding precision-driven false negatives.
        """
        s = LandmarkSmoother()
        threshold = LandmarkSmoother.IDENTITY_THRESHOLD
        # Nudge slightly above threshold to survive float32 rounding
        sim = threshold + 1e-4
        perpendicular = (1.0 - sim ** 2) ** 0.5
        emb = np.array([sim, perpendicular, 0.0], dtype=np.float32)
        emb /= np.linalg.norm(emb)
        # Verify the cosine is indeed >= threshold after float32 normalisation
        assert float(np.array([1.0, 0.0, 0.0], dtype=np.float32) @ emb) >= threshold
        entry = {'embedding': _unit(_EMB_X), 'bbox': None, 'kps': None}
        s._state = [entry]
        face = _make_face(emb)
        result = s._find_match(face)
        assert result is entry


# ---------------------------------------------------------------------------
# _find_match: best-match selection with multiple state entries
# ---------------------------------------------------------------------------

class TestFindMatchMultipleEntries:
    def test_returns_best_matching_entry(self):
        """With multiple state entries, the one with highest cosine is returned."""
        s = LandmarkSmoother()
        entry_x = {'embedding': _unit(_EMB_X), 'bbox': np.array([0.0]), 'kps': None}
        entry_y = {'embedding': _unit(_EMB_Y), 'bbox': np.array([1.0]), 'kps': None}
        s._state = [entry_x, entry_y]

        # Face most similar to X → should return entry_x
        face = _make_face(_EMB_X)
        result = s._find_match(face)
        assert result is entry_x

    def test_returns_second_entry_when_it_is_best(self):
        """Symmetry check: entry_y wins when face embedding is closest to Y."""
        s = LandmarkSmoother()
        entry_x = {'embedding': _unit(_EMB_X), 'bbox': np.array([0.0]), 'kps': None}
        entry_y = {'embedding': _unit(_EMB_Y), 'bbox': np.array([1.0]), 'kps': None}
        s._state = [entry_x, entry_y]

        face = _make_face(_EMB_Y)
        result = s._find_match(face)
        assert result is entry_y

    def test_three_entries_selects_correct_one(self):
        """Three orthogonal entries — each face embedding should select its own."""
        s = LandmarkSmoother()
        entry_x = {'embedding': _unit(_EMB_X), 'bbox': None, 'kps': None}
        entry_y = {'embedding': _unit(_EMB_Y), 'bbox': None, 'kps': None}
        entry_z = {'embedding': _unit(_EMB_Z), 'bbox': None, 'kps': None}
        s._state = [entry_x, entry_y, entry_z]

        assert s._find_match(_make_face(_EMB_X)) is entry_x
        assert s._find_match(_make_face(_EMB_Y)) is entry_y
        assert s._find_match(_make_face(_EMB_Z)) is entry_z

    def test_all_below_threshold_returns_none(self):
        """If all stored identities are dissimilar, None is returned even with many entries."""
        s = LandmarkSmoother()
        entry_y = {'embedding': _unit(_EMB_Y), 'bbox': None, 'kps': None}
        entry_z = {'embedding': _unit(_EMB_Z), 'bbox': None, 'kps': None}
        s._state = [entry_y, entry_z]

        # Face is unit-x → cosine with Y = 0, cosine with Z = 0, both < threshold
        face = _make_face(_EMB_X)
        assert s._find_match(face) is None

    def test_partial_above_threshold_selects_best(self):
        """Only one entry above threshold — that one should be returned."""
        s = LandmarkSmoother()
        # entry_close has cosine ≈ 0.99 to _EMB_X (above threshold)
        close = np.array([0.99, 0.14, 0.0], dtype=np.float32)
        close /= np.linalg.norm(close)
        entry_close = {'embedding': close, 'bbox': None, 'kps': None}

        # entry_far is orthogonal (cosine = 0, below threshold)
        entry_far = {'embedding': _unit(_EMB_Y), 'bbox': None, 'kps': None}

        s._state = [entry_far, entry_close]
        face = _make_face(_EMB_X)
        result = s._find_match(face)
        assert result is entry_close
