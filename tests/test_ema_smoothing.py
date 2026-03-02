"""Unit tests for LandmarkSmoother EMA smoothing (Issue #67).

Tests cover:
- EMA arithmetic correctness
- Identity matching via embedding cosine similarity
- Face count changes and reset behaviour
- None / missing attribute handling
- Convergence over multiple frames
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from modules.face_analyser import LandmarkSmoother


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_face(bbox=None, kps=None, embedding=None):
    """Return a mock face object with the given attributes."""
    face = MagicMock()
    face.bbox = np.array(bbox, dtype=np.float32) if bbox is not None else None
    face.kps = np.array(kps, dtype=np.float32) if kps is not None else None
    face.normed_embedding = (
        np.array(embedding, dtype=np.float32) if embedding is not None else None
    )
    return face


def _unit_embedding(ndim: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(ndim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_detection_result(target_face=None, many_faces=None):
    return {'target_face': target_face, 'many_faces': many_faces}


# ---------------------------------------------------------------------------
# Constructor / parameter validation
# ---------------------------------------------------------------------------

class TestLandmarkSmootherInit:
    def test_default_alpha(self):
        s = LandmarkSmoother()
        assert s.alpha == 0.7

    def test_custom_alpha(self):
        s = LandmarkSmoother(alpha=0.9)
        assert s.alpha == 0.9

    def test_alpha_boundary_one(self):
        s = LandmarkSmoother(alpha=1.0)
        assert s.alpha == 1.0

    def test_alpha_zero_invalid(self):
        with pytest.raises(ValueError):
            LandmarkSmoother(alpha=0.0)

    def test_alpha_negative_invalid(self):
        with pytest.raises(ValueError):
            LandmarkSmoother(alpha=-0.1)

    def test_alpha_above_one_invalid(self):
        with pytest.raises(ValueError):
            LandmarkSmoother(alpha=1.1)


# ---------------------------------------------------------------------------
# Basic EMA arithmetic
# ---------------------------------------------------------------------------

class TestEMASingleFace:
    """Verify EMA formula: smoothed = alpha * current + (1-alpha) * prev."""

    def test_first_frame_no_smoothing(self):
        """First detection: output equals input (no history)."""
        s = LandmarkSmoother(alpha=0.7)
        emb = _unit_embedding(seed=1)
        bbox = [10.0, 20.0, 100.0, 120.0]
        face = _make_face(bbox=bbox, embedding=emb)
        result = _make_detection_result(target_face=face)
        s.smooth(result)
        np.testing.assert_allclose(face.bbox, np.array(bbox, dtype=np.float32))

    def test_second_frame_ema_applied_to_bbox(self):
        s = LandmarkSmoother(alpha=0.7)
        emb = _unit_embedding(seed=1)
        bbox1 = [10.0, 20.0, 100.0, 120.0]
        bbox2 = [12.0, 22.0, 102.0, 122.0]

        face1 = _make_face(bbox=bbox1, embedding=emb)
        face2 = _make_face(bbox=bbox2, embedding=emb)

        s.smooth(_make_detection_result(target_face=face1))
        s.smooth(_make_detection_result(target_face=face2))

        expected = 0.7 * np.array(bbox2, dtype=np.float32) + 0.3 * np.array(bbox1, dtype=np.float32)
        np.testing.assert_allclose(face2.bbox, expected, rtol=1e-5)

    def test_second_frame_ema_applied_to_kps(self):
        s = LandmarkSmoother(alpha=0.7)
        emb = _unit_embedding(seed=2)
        kps1 = [[40.0, 50.0], [60.0, 50.0], [50.0, 70.0]]
        kps2 = [[42.0, 51.0], [62.0, 51.0], [52.0, 71.0]]

        face1 = _make_face(kps=kps1, embedding=emb)
        face2 = _make_face(kps=kps2, embedding=emb)

        s.smooth(_make_detection_result(target_face=face1))
        s.smooth(_make_detection_result(target_face=face2))

        expected = 0.7 * np.array(kps2, dtype=np.float32) + 0.3 * np.array(kps1, dtype=np.float32)
        np.testing.assert_allclose(face2.kps, expected, rtol=1e-5)

    def test_alpha_one_means_no_smoothing(self):
        """alpha=1.0 should pass through the current frame unchanged."""
        s = LandmarkSmoother(alpha=1.0)
        emb = _unit_embedding(seed=3)
        bbox1 = [10.0, 20.0, 100.0, 120.0]
        bbox2 = [50.0, 60.0, 150.0, 160.0]

        face1 = _make_face(bbox=bbox1, embedding=emb)
        face2 = _make_face(bbox=bbox2, embedding=emb)

        s.smooth(_make_detection_result(target_face=face1))
        s.smooth(_make_detection_result(target_face=face2))

        np.testing.assert_allclose(face2.bbox, np.array(bbox2, dtype=np.float32))


# ---------------------------------------------------------------------------
# Identity matching
# ---------------------------------------------------------------------------

class TestIdentityMatching:
    def test_same_identity_smoothed(self):
        """Same embedding → state is reused and EMA applied."""
        s = LandmarkSmoother(alpha=0.5)
        emb = _unit_embedding(seed=10)
        # Slightly perturb embedding to simulate realistic frame-to-frame variation
        emb2 = emb + 0.001 * _unit_embedding(seed=99)
        emb2 = emb2 / np.linalg.norm(emb2)

        face1 = _make_face(bbox=[0.0, 0.0, 100.0, 100.0], embedding=emb)
        face2 = _make_face(bbox=[10.0, 10.0, 110.0, 110.0], embedding=emb2)

        s.smooth(_make_detection_result(target_face=face1))
        s.smooth(_make_detection_result(target_face=face2))

        # EMA should be applied: bbox != current raw detection
        expected = 0.5 * np.array([10, 10, 110, 110], dtype=np.float32) + 0.5 * np.array([0, 0, 100, 100], dtype=np.float32)
        np.testing.assert_allclose(face2.bbox, expected, rtol=1e-5)

    def test_different_identity_resets_state(self):
        """A new identity (low cosine) gets fresh state, no EMA from previous."""
        s = LandmarkSmoother(alpha=0.7)
        emb_a = _unit_embedding(seed=11)
        emb_b = _unit_embedding(seed=22)
        # Ensure low cosine similarity between the two identities
        cos = float(np.dot(emb_a, emb_b))
        assert abs(cos) < LandmarkSmoother.IDENTITY_COSINE_THRESHOLD, (
            f"Test embeddings unexpectedly similar: cos={cos}"
        )

        face1 = _make_face(bbox=[0.0, 0.0, 100.0, 100.0], embedding=emb_a)
        face2 = _make_face(bbox=[200.0, 200.0, 300.0, 300.0], embedding=emb_b)

        s.smooth(_make_detection_result(target_face=face1))
        s.smooth(_make_detection_result(target_face=face2))

        # face2 is a new identity — should keep its raw bbox unchanged
        np.testing.assert_allclose(face2.bbox, np.array([200.0, 200.0, 300.0, 300.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# Many-faces mode
# ---------------------------------------------------------------------------

class TestManyFaces:
    def test_multiple_faces_each_smoothed(self):
        """Each face in many_faces gets its own EMA state."""
        s = LandmarkSmoother(alpha=0.6)
        emb_a = _unit_embedding(seed=30)
        emb_b = _unit_embedding(seed=31)

        face_a1 = _make_face(bbox=[0, 0, 50, 50], embedding=emb_a)
        face_b1 = _make_face(bbox=[100, 100, 150, 150], embedding=emb_b)
        face_a2 = _make_face(bbox=[5, 5, 55, 55], embedding=emb_a)
        face_b2 = _make_face(bbox=[105, 105, 155, 155], embedding=emb_b)

        s.smooth(_make_detection_result(many_faces=[face_a1, face_b1]))
        s.smooth(_make_detection_result(many_faces=[face_a2, face_b2]))

        expected_a = 0.6 * np.array([5, 5, 55, 55], dtype=np.float32) + 0.4 * np.array([0, 0, 50, 50], dtype=np.float32)
        expected_b = 0.6 * np.array([105, 105, 155, 155], dtype=np.float32) + 0.4 * np.array([100, 100, 150, 150], dtype=np.float32)

        np.testing.assert_allclose(face_a2.bbox, expected_a, rtol=1e-5)
        np.testing.assert_allclose(face_b2.bbox, expected_b, rtol=1e-5)

    def test_face_count_change_clears_state(self):
        """When no faces detected, state resets so next frame starts fresh."""
        s = LandmarkSmoother(alpha=0.7)
        emb = _unit_embedding(seed=40)
        face1 = _make_face(bbox=[0, 0, 100, 100], embedding=emb)
        face2 = _make_face(bbox=[10, 10, 110, 110], embedding=emb)

        s.smooth(_make_detection_result(target_face=face1))
        # No face detected — should clear state
        s.smooth(_make_detection_result(target_face=None, many_faces=None))
        # Next frame should behave like first-seen
        s.smooth(_make_detection_result(target_face=face2))

        # face2 treated as fresh — no EMA from face1
        np.testing.assert_allclose(face2.bbox, np.array([10, 10, 110, 110], dtype=np.float32))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_face_with_no_embedding_tracked(self):
        """Face without embedding still gets bbox recorded (no identity matching)."""
        s = LandmarkSmoother(alpha=0.7)
        face1 = _make_face(bbox=[0, 0, 100, 100], embedding=None)
        face2 = _make_face(bbox=[5, 5, 105, 105], embedding=None)

        s.smooth(_make_detection_result(target_face=face1))
        s.smooth(_make_detection_result(target_face=face2))

        # No matching (no embedding) → face2 stays as-is (new identity created each time)
        np.testing.assert_allclose(face2.bbox, np.array([5, 5, 105, 105], dtype=np.float32))

    def test_face_with_none_bbox_handled(self):
        """Face with None bbox should not raise."""
        s = LandmarkSmoother(alpha=0.7)
        emb = _unit_embedding(seed=50)
        face = _make_face(bbox=None, embedding=emb)
        result = _make_detection_result(target_face=face)
        s.smooth(result)  # should not raise

    def test_empty_result_clears_state(self):
        """Completely empty result clears smoother state."""
        s = LandmarkSmoother(alpha=0.7)
        emb = _unit_embedding(seed=60)
        face = _make_face(bbox=[0, 0, 100, 100], embedding=emb)
        s.smooth(_make_detection_result(target_face=face))
        s.smooth(_make_detection_result(target_face=None, many_faces=[]))
        assert s._states == []

    def test_reset_clears_all_state(self):
        s = LandmarkSmoother(alpha=0.7)
        emb = _unit_embedding(seed=70)
        face = _make_face(bbox=[0, 0, 100, 100], embedding=emb)
        s.smooth(_make_detection_result(target_face=face))
        assert len(s._states) > 0
        s.reset()
        assert s._states == []

    def test_smooth_target_and_many_faces_combined(self):
        """Result dict can have both target_face and many_faces populated."""
        s = LandmarkSmoother(alpha=0.8)
        emb_a = _unit_embedding(seed=80)
        emb_b = _unit_embedding(seed=81)

        face_a = _make_face(bbox=[0, 0, 50, 50], embedding=emb_a)
        face_b = _make_face(bbox=[100, 100, 150, 150], embedding=emb_b)
        result = {'target_face': face_a, 'many_faces': [face_b]}
        s.smooth(result)  # should not raise and both faces recorded
        assert len(s._states) == 2


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_converges_to_stable_position(self):
        """After many frames at the same position, output converges to that position."""
        s = LandmarkSmoother(alpha=0.3)
        emb = _unit_embedding(seed=90)
        stable_bbox = [50.0, 50.0, 150.0, 150.0]

        for _ in range(50):
            face = _make_face(bbox=stable_bbox, embedding=emb)
            s.smooth(_make_detection_result(target_face=face))

        np.testing.assert_allclose(face.bbox, np.array(stable_bbox, dtype=np.float32), atol=1e-3)

    def test_heavy_smoothing_lags_behind_fast_movement(self):
        """With low alpha (heavy smoothing), output should lag the current position."""
        s = LandmarkSmoother(alpha=0.1)
        emb = _unit_embedding(seed=95)

        # Start at origin
        face0 = _make_face(bbox=[0, 0, 100, 100], embedding=emb)
        s.smooth(_make_detection_result(target_face=face0))

        # Jump to far-away position
        face1 = _make_face(bbox=[500, 500, 600, 600], embedding=emb)
        s.smooth(_make_detection_result(target_face=face1))

        # Output should be much closer to origin than to the new position
        assert float(face1.bbox[0]) < 100.0, "Heavy smoothing should lag behind fast movement"
