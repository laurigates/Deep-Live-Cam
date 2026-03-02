"""Unit tests for LandmarkSmoother EMA smoothing (issue #67).

RED → GREEN workflow: these tests define the expected behaviour of the
LandmarkSmoother class added to modules/face_analyser.py.
"""
import numpy as np
import pytest

from modules.face_analyser import LandmarkSmoother


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_face(bbox, kps=None, embedding=None):
    """Minimal InsightFace-like face object (dict subclass)."""

    class _Face(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

        def __setattr__(self, name, value):
            self[name] = value

    f = _Face()
    f.bbox = np.array(bbox, dtype=np.float32) if bbox is not None else None
    f.kps = np.array(kps, dtype=np.float32) if kps is not None else None
    if embedding is not None:
        raw = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(raw)
        f.normed_embedding = raw / norm if norm > 0 else raw
    else:
        f.normed_embedding = None
    return f


_BBOX_A = [10.0, 20.0, 110.0, 120.0]
_BBOX_B = [200.0, 200.0, 300.0, 300.0]  # far from A
_KPS_A = [[30.0, 40.0], [70.0, 40.0], [50.0, 60.0], [35.0, 80.0], [65.0, 80.0]]
_EMB_A = [1.0, 0.0, 0.0]   # unit vector along x
_EMB_B = [0.0, 1.0, 0.0]   # orthogonal — different identity


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestLandmarkSmootherInit:
    def test_default_alpha(self):
        s = LandmarkSmoother()
        assert s.alpha == pytest.approx(0.7)

    def test_custom_alpha(self):
        s = LandmarkSmoother(alpha=0.5)
        assert s.alpha == pytest.approx(0.5)

    def test_alpha_clamped_to_zero(self):
        s = LandmarkSmoother(alpha=-0.5)
        assert s.alpha == pytest.approx(0.0)

    def test_alpha_clamped_to_one(self):
        s = LandmarkSmoother(alpha=1.5)
        assert s.alpha == pytest.approx(1.0)

    def test_state_initially_empty(self):
        s = LandmarkSmoother()
        assert s._state == []


# ---------------------------------------------------------------------------
# alpha property setter
# ---------------------------------------------------------------------------

class TestAlphaSetter:
    def test_setter_updates_alpha(self):
        s = LandmarkSmoother(alpha=0.7)
        s.alpha = 0.9
        assert s.alpha == pytest.approx(0.9)

    def test_setter_clamps_below_zero(self):
        s = LandmarkSmoother()
        s.alpha = -1.0
        assert s.alpha == pytest.approx(0.0)

    def test_setter_clamps_above_one(self):
        s = LandmarkSmoother()
        s.alpha = 2.0
        assert s.alpha == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# smooth() — empty / None input
# ---------------------------------------------------------------------------

class TestSmoothEmpty:
    def test_empty_list_returns_empty(self):
        s = LandmarkSmoother()
        result = s.smooth([])
        assert result == []

    def test_empty_list_clears_state(self):
        s = LandmarkSmoother()
        face = _make_face(_BBOX_A, embedding=_EMB_A)
        s.smooth([face])          # populate state
        s.smooth([])              # now clear it
        assert s._state == []

    def test_returns_same_list_object(self):
        s = LandmarkSmoother()
        faces = [_make_face(_BBOX_A, embedding=_EMB_A)]
        returned = s.smooth(faces)
        assert returned is faces


# ---------------------------------------------------------------------------
# smooth() — first frame (no previous state)
# ---------------------------------------------------------------------------

class TestSmoothFirstFrame:
    def test_bbox_unchanged_on_first_frame(self):
        s = LandmarkSmoother(alpha=0.7)
        face = _make_face(_BBOX_A, embedding=_EMB_A)
        original_bbox = face.bbox.copy()
        s.smooth([face])
        np.testing.assert_array_almost_equal(face.bbox, original_bbox)

    def test_kps_unchanged_on_first_frame(self):
        s = LandmarkSmoother(alpha=0.7)
        face = _make_face(_BBOX_A, kps=_KPS_A, embedding=_EMB_A)
        original_kps = face.kps.copy()
        s.smooth([face])
        np.testing.assert_array_almost_equal(face.kps, original_kps)

    def test_state_populated_after_first_frame(self):
        s = LandmarkSmoother()
        face = _make_face(_BBOX_A, embedding=_EMB_A)
        s.smooth([face])
        assert len(s._state) == 1
        np.testing.assert_array_almost_equal(s._state[0]['bbox'], _BBOX_A)


# ---------------------------------------------------------------------------
# smooth() — EMA blending on second frame (same identity)
# ---------------------------------------------------------------------------

class TestSmoothEmaBlending:
    def _run_two_frames(self, alpha, bbox1, bbox2, emb):
        s = LandmarkSmoother(alpha=alpha)
        face1 = _make_face(bbox1, embedding=emb)
        s.smooth([face1])
        face2 = _make_face(bbox2, embedding=emb)
        s.smooth([face2])
        return face2

    def test_ema_bbox_blending(self):
        alpha = 0.7
        bbox1 = [0.0, 0.0, 100.0, 100.0]
        bbox2 = [10.0, 10.0, 110.0, 110.0]
        expected = np.array(bbox2) * alpha + np.array(bbox1) * (1 - alpha)
        face2 = self._run_two_frames(alpha, bbox1, bbox2, _EMB_A)
        np.testing.assert_array_almost_equal(face2.bbox, expected)

    def test_alpha_one_no_smoothing(self):
        """alpha=1.0 means no blending — output equals current detection."""
        bbox1 = [0.0, 0.0, 100.0, 100.0]
        bbox2 = [50.0, 50.0, 150.0, 150.0]
        face2 = self._run_two_frames(1.0, bbox1, bbox2, _EMB_A)
        np.testing.assert_array_almost_equal(face2.bbox, bbox2)

    def test_alpha_zero_full_history(self):
        """alpha=0.0 means fully replace output with history (no current)."""
        bbox1 = [0.0, 0.0, 100.0, 100.0]
        bbox2 = [50.0, 50.0, 150.0, 150.0]
        face2 = self._run_two_frames(0.0, bbox1, bbox2, _EMB_A)
        np.testing.assert_array_almost_equal(face2.bbox, bbox1)

    def test_ema_kps_blending(self):
        alpha = 0.6
        kps1 = [[10.0, 20.0], [30.0, 40.0]]
        kps2 = [[20.0, 30.0], [40.0, 50.0]]
        expected = np.array(kps2) * alpha + np.array(kps1) * (1 - alpha)

        s = LandmarkSmoother(alpha=alpha)
        face1 = _make_face(_BBOX_A, kps=kps1, embedding=_EMB_A)
        s.smooth([face1])
        face2 = _make_face(_BBOX_A, kps=kps2, embedding=_EMB_A)
        s.smooth([face2])

        np.testing.assert_array_almost_equal(face2.kps, expected)


# ---------------------------------------------------------------------------
# smooth() — identity mismatch → no blending (new face)
# ---------------------------------------------------------------------------

class TestIdentityReset:
    def test_different_identity_no_blending(self):
        """When cosine similarity < IDENTITY_THRESHOLD, bbox should not be blended."""
        s = LandmarkSmoother(alpha=0.7)
        # Frame 1: identity A
        face1 = _make_face(_BBOX_A, embedding=_EMB_A)
        s.smooth([face1])

        # Frame 2: identity B (orthogonal embedding → cosine = 0 < 0.7)
        face2 = _make_face(_BBOX_B, embedding=_EMB_B)
        original_bbox2 = face2.bbox.copy()
        s.smooth([face2])

        # bbox should be unchanged (raw detection, no blending with A's position)
        np.testing.assert_array_almost_equal(face2.bbox, original_bbox2)

    def test_state_holds_new_identity_after_change(self):
        """After identity change, state should store the new face, not the old one."""
        s = LandmarkSmoother(alpha=0.7)
        face1 = _make_face(_BBOX_A, embedding=_EMB_A)
        s.smooth([face1])
        face2 = _make_face(_BBOX_B, embedding=_EMB_B)
        s.smooth([face2])

        # State should now reflect face2 (B)
        assert len(s._state) == 1
        np.testing.assert_array_almost_equal(s._state[0]['bbox'], _BBOX_B)


# ---------------------------------------------------------------------------
# smooth() — face with no embedding
# ---------------------------------------------------------------------------

class TestNoEmbedding:
    def test_no_embedding_first_frame_bbox_unchanged(self):
        s = LandmarkSmoother(alpha=0.7)
        face = _make_face(_BBOX_A)  # no embedding
        original = face.bbox.copy()
        s.smooth([face])
        np.testing.assert_array_almost_equal(face.bbox, original)

    def test_no_embedding_second_frame_no_blending(self):
        """Without embeddings, we cannot match identities, so no blending occurs."""
        s = LandmarkSmoother(alpha=0.7)
        face1 = _make_face([0.0, 0.0, 100.0, 100.0])
        s.smooth([face1])

        face2 = _make_face([50.0, 50.0, 150.0, 150.0])
        original2 = face2.bbox.copy()
        s.smooth([face2])

        # No match found (no embeddings) → bbox unchanged
        np.testing.assert_array_almost_equal(face2.bbox, original2)


# ---------------------------------------------------------------------------
# smooth() — None bbox / kps handling
# ---------------------------------------------------------------------------

class TestNullCoordinates:
    def test_none_bbox_does_not_crash(self):
        s = LandmarkSmoother()
        face1 = _make_face(None, embedding=_EMB_A)
        s.smooth([face1])   # should not raise

    def test_none_kps_does_not_crash(self):
        s = LandmarkSmoother()
        face1 = _make_face(_BBOX_A, kps=None, embedding=_EMB_A)
        face2 = _make_face(_BBOX_A, kps=None, embedding=_EMB_A)
        s.smooth([face1])
        s.smooth([face2])   # should not raise


# ---------------------------------------------------------------------------
# smooth() — multiple faces
# ---------------------------------------------------------------------------

class TestMultipleFaces:
    def test_two_faces_independently_smoothed(self):
        """Each face should be matched to its own identity state."""
        alpha = 0.8
        s = LandmarkSmoother(alpha=alpha)

        bbox_a1 = [0.0, 0.0, 50.0, 50.0]
        bbox_b1 = [100.0, 100.0, 150.0, 150.0]
        face_a1 = _make_face(bbox_a1, embedding=_EMB_A)
        face_b1 = _make_face(bbox_b1, embedding=_EMB_B)
        s.smooth([face_a1, face_b1])

        bbox_a2 = [5.0, 5.0, 55.0, 55.0]
        bbox_b2 = [105.0, 105.0, 155.0, 155.0]
        face_a2 = _make_face(bbox_a2, embedding=_EMB_A)
        face_b2 = _make_face(bbox_b2, embedding=_EMB_B)
        s.smooth([face_a2, face_b2])

        exp_a = np.array(bbox_a2) * alpha + np.array(bbox_a1) * (1 - alpha)
        exp_b = np.array(bbox_b2) * alpha + np.array(bbox_b1) * (1 - alpha)
        np.testing.assert_array_almost_equal(face_a2.bbox, exp_a)
        np.testing.assert_array_almost_equal(face_b2.bbox, exp_b)

    def test_face_count_drop_does_not_corrupt_state(self):
        """Losing a face between frames should not cause an error or wrong matches."""
        s = LandmarkSmoother(alpha=0.7)
        face_a = _make_face(_BBOX_A, embedding=_EMB_A)
        face_b = _make_face(_BBOX_B, embedding=_EMB_B)
        s.smooth([face_a, face_b])

        # Only face_a in next frame
        face_a2 = _make_face(_BBOX_A, embedding=_EMB_A)
        s.smooth([face_a2])   # should not raise
        assert len(s._state) == 1

    def test_new_face_appears(self):
        """A face with no previous match should receive no smoothing."""
        s = LandmarkSmoother(alpha=0.7)
        face_a = _make_face(_BBOX_A, embedding=_EMB_A)
        s.smooth([face_a])

        # Frame 2: face_a + new face_b
        face_a2 = _make_face(_BBOX_A, embedding=_EMB_A)
        face_b2 = _make_face(_BBOX_B, embedding=_EMB_B)
        original_b = face_b2.bbox.copy()
        s.smooth([face_a2, face_b2])

        # face_b2 is new — its bbox must not be blended with face_a's position
        np.testing.assert_array_almost_equal(face_b2.bbox, original_b)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_state(self):
        s = LandmarkSmoother()
        face = _make_face(_BBOX_A, embedding=_EMB_A)
        s.smooth([face])
        assert len(s._state) == 1
        s.reset()
        assert s._state == []

    def test_reset_then_smooth_gives_fresh_start(self):
        """After reset, the next smooth call should behave like the first frame."""
        alpha = 0.7
        s = LandmarkSmoother(alpha=alpha)
        face1 = _make_face([0.0, 0.0, 100.0, 100.0], embedding=_EMB_A)
        s.smooth([face1])
        s.reset()

        face2 = _make_face([50.0, 50.0, 150.0, 150.0], embedding=_EMB_A)
        original2 = face2.bbox.copy()
        s.smooth([face2])

        # No prior state → no blending → bbox should match the raw detection
        np.testing.assert_array_almost_equal(face2.bbox, original2)


# ---------------------------------------------------------------------------
# Accumulation across many frames
# ---------------------------------------------------------------------------

class TestAccumulation:
    def test_repeated_smoothing_converges(self):
        """Over many frames with the same position, smoothed bbox → static position."""
        s = LandmarkSmoother(alpha=0.7)
        target_bbox = [50.0, 50.0, 150.0, 150.0]
        emb = _EMB_A

        for _ in range(30):
            face = _make_face(target_bbox, embedding=emb)
            s.smooth([face])

        # After 30 frames of identical input, smoothed bbox should be very close
        face = _make_face(target_bbox, embedding=emb)
        s.smooth([face])
        np.testing.assert_array_almost_equal(face.bbox, target_bbox, decimal=3)
