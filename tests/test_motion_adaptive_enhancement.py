"""Tests for motion-adaptive enhancement frequency (issue #50).

RED phase: tests are written before the implementation so they initially fail,
then pass once the utilities are added to face_analyser.py.
"""

import numpy as np
import pytest

from modules.face_analyser import (
    compute_bbox_iou,
    compute_embedding_cosine,
    faces_are_similar,
)

# ---------------------------------------------------------------------------
# compute_bbox_iou
# ---------------------------------------------------------------------------


class TestComputeBboxIou:
    def test_identical_boxes_returns_one(self):
        bbox = np.array([10.0, 20.0, 100.0, 200.0])
        assert compute_bbox_iou(bbox, bbox) == pytest.approx(1.0)

    def test_non_overlapping_returns_zero(self):
        a = np.array([0.0, 0.0, 10.0, 10.0])
        b = np.array([20.0, 20.0, 30.0, 30.0])
        assert compute_bbox_iou(a, b) == pytest.approx(0.0)

    def test_half_overlap(self):
        # Two 10x10 boxes sharing a 10x5 strip → IoU = 50/150
        a = np.array([0.0, 0.0, 10.0, 10.0])
        b = np.array([0.0, 5.0, 10.0, 15.0])
        iou = compute_bbox_iou(a, b)
        assert iou == pytest.approx(50.0 / 150.0, rel=1e-5)

    def test_one_inside_other(self):
        outer = np.array([0.0, 0.0, 20.0, 20.0])  # area 400
        inner = np.array([5.0, 5.0, 15.0, 15.0])  # area 100
        # intersection = 100, union = 400
        iou = compute_bbox_iou(outer, inner)
        assert iou == pytest.approx(100.0 / 400.0, rel=1e-5)

    def test_touching_edge_no_area_overlap(self):
        a = np.array([0.0, 0.0, 10.0, 10.0])
        b = np.array([10.0, 0.0, 20.0, 10.0])
        assert compute_bbox_iou(a, b) == pytest.approx(0.0)

    def test_handles_integer_arrays(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([0, 0, 10, 10])
        assert compute_bbox_iou(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_embedding_cosine
# ---------------------------------------------------------------------------


class TestComputeEmbeddingCosine:
    def test_identical_embeddings_returns_one(self):
        emb = np.array([0.6, 0.8], dtype=np.float32)  # already L2-normed
        assert compute_embedding_cosine(emb, emb) == pytest.approx(1.0)

    def test_orthogonal_embeddings_returns_zero(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert compute_embedding_cosine(a, b) == pytest.approx(0.0)

    def test_opposite_embeddings_returns_minus_one(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert compute_embedding_cosine(a, b) == pytest.approx(-1.0)

    def test_partial_similarity(self):
        # 45° apart on the unit circle
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)], dtype=np.float32)
        expected = np.cos(np.pi / 4)
        assert compute_embedding_cosine(a, b) == pytest.approx(expected, rel=1e-5)

    def test_returns_float(self):
        emb = np.ones(512, dtype=np.float32) / np.sqrt(512)
        result = compute_embedding_cosine(emb, emb)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# faces_are_similar
# ---------------------------------------------------------------------------


def _make_face(bbox, embedding=None):
    """Minimal face-like object with bbox and normed_embedding."""

    class _Face:
        pass

    f = _Face()
    f.bbox = np.array(bbox, dtype=np.float32)
    if embedding is not None:
        f.normed_embedding = np.array(embedding, dtype=np.float32)
    else:
        f.normed_embedding = None
    return f


_IDENTICAL_BBOX = [10.0, 20.0, 110.0, 120.0]
_IDENTICAL_EMB = [0.6, 0.8]  # unit vector
_MOVED_BBOX = [50.0, 60.0, 150.0, 160.0]  # IoU well below 0.9
_DIFFERENT_EMB = [0.0, 1.0]  # cosine = 0 with [0.6, 0.8] → below 0.95


class TestFacesAreSimilar:
    def test_single_identical_face_is_similar(self):
        face = _make_face(_IDENTICAL_BBOX, _IDENTICAL_EMB)
        prev = _make_face(_IDENTICAL_BBOX, _IDENTICAL_EMB)
        assert faces_are_similar([face], [prev]) is True

    def test_empty_current_faces_returns_false(self):
        prev = _make_face(_IDENTICAL_BBOX, _IDENTICAL_EMB)
        assert faces_are_similar([], [prev]) is False

    def test_empty_prev_faces_returns_false(self):
        face = _make_face(_IDENTICAL_BBOX, _IDENTICAL_EMB)
        assert faces_are_similar([face], []) is False

    def test_none_prev_faces_returns_false(self):
        face = _make_face(_IDENTICAL_BBOX, _IDENTICAL_EMB)
        assert faces_are_similar([face], None) is False

    def test_none_current_faces_returns_false(self):
        prev = _make_face(_IDENTICAL_BBOX, _IDENTICAL_EMB)
        assert faces_are_similar(None, [prev]) is False

    def test_different_face_count_returns_false(self):
        face = _make_face(_IDENTICAL_BBOX, _IDENTICAL_EMB)
        prev1 = _make_face(_IDENTICAL_BBOX, _IDENTICAL_EMB)
        prev2 = _make_face(_IDENTICAL_BBOX, _IDENTICAL_EMB)
        assert faces_are_similar([face], [prev1, prev2]) is False

    def test_moved_face_returns_false(self):
        # bbox IoU well below default 0.9
        face = _make_face(_MOVED_BBOX, _IDENTICAL_EMB)
        prev = _make_face(_IDENTICAL_BBOX, _IDENTICAL_EMB)
        assert faces_are_similar([face], [prev]) is False

    def test_changed_embedding_returns_false(self):
        face = _make_face(_IDENTICAL_BBOX, _DIFFERENT_EMB)
        prev = _make_face(_IDENTICAL_BBOX, _IDENTICAL_EMB)
        assert faces_are_similar([face], [prev]) is False

    def test_custom_thresholds_accepted(self):
        # Lower IoU threshold so a slightly-moved bbox still qualifies
        face = _make_face([10.0, 20.0, 110.0, 120.0], _IDENTICAL_EMB)
        # Shift bbox by 1px
        prev = _make_face([11.0, 21.0, 111.0, 121.0], _IDENTICAL_EMB)
        iou = compute_bbox_iou(face.bbox, prev.bbox)
        assert iou > 0.95  # sanity check that shift is tiny
        assert faces_are_similar([face], [prev], iou_threshold=0.9) is True

    def test_face_missing_bbox_returns_false(self):
        face = _make_face(_IDENTICAL_BBOX, _IDENTICAL_EMB)
        face.bbox = None
        prev = _make_face(_IDENTICAL_BBOX, _IDENTICAL_EMB)
        assert faces_are_similar([face], [prev]) is False

    def test_face_no_normed_embedding_still_similar_if_bbox_ok(self):
        # If embedding is absent on both, only bbox check applies
        face = _make_face(_IDENTICAL_BBOX)  # normed_embedding = None
        prev = _make_face(_IDENTICAL_BBOX)
        # Both lack embedding → embedding check is skipped → pass if bbox matches
        assert faces_are_similar([face], [prev]) is True

    def test_multiple_identical_faces_similar(self):
        faces = [_make_face(_IDENTICAL_BBOX, _IDENTICAL_EMB) for _ in range(3)]
        prevs = [_make_face(_IDENTICAL_BBOX, _IDENTICAL_EMB) for _ in range(3)]
        assert faces_are_similar(faces, prevs) is True

    def test_multiple_faces_one_moved(self):
        faces = [
            _make_face(_IDENTICAL_BBOX, _IDENTICAL_EMB),
            _make_face(_MOVED_BBOX, _IDENTICAL_EMB),  # this one moved
        ]
        prevs = [
            _make_face(_IDENTICAL_BBOX, _IDENTICAL_EMB),
            _make_face(_IDENTICAL_BBOX, _IDENTICAL_EMB),
        ]
        assert faces_are_similar(faces, prevs) is False
