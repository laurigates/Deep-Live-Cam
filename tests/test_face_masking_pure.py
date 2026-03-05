"""Unit tests for pure / near-pure functions in face_masking.py.

Tests create_lower_mouth_mask edge cases, _eye_dimensions, _ellipse_polygon,
_curved_eyebrow_contour, and apply_mask_area boundary conditions.
"""

from types import SimpleNamespace

import numpy as np


def _make_face_with_landmarks(frame_h=480, frame_w=640):
    """Create a mock face with realistic 106-point landmarks."""
    face = SimpleNamespace()

    lm = np.zeros((106, 2), dtype=np.float32)
    cx, cy = frame_w / 2, frame_h / 2

    # Face outline (0-32)
    for i in range(33):
        angle = np.pi * i / 32
        lm[i] = [cx + 90 * np.sin(angle), cy - 120 * np.cos(angle)]

    # Left eyebrow (33-42)
    for i, idx in enumerate(range(33, 43)):
        lm[idx] = [cx - 50 + i * 8, cy - 80 - abs(i - 5) * 2]

    # Right eyebrow (43-51)
    for i, idx in enumerate(range(43, 52)):
        lm[idx] = [cx + 10 + i * 8, cy - 80 - abs(i - 4) * 2]

    # Lower lip (52-63)
    for i, idx in enumerate(range(52, 64)):
        angle = 2 * np.pi * i / 12
        lm[idx] = [cx + 25 * np.cos(angle), cy + 60 + 10 * np.sin(angle)]

    # Nose (64-72)
    for i, idx in enumerate(range(64, 73)):
        lm[idx] = [cx - 10 + i * 2.5, cy - 20 + abs(i - 4) * 3]

    # Left eye (73-86)
    for i, idx in enumerate(range(73, 87)):
        angle = 2 * np.pi * i / 14
        lm[idx] = [cx - 30 + 15 * np.cos(angle), cy - 40 + 8 * np.sin(angle)]

    # Right eye (87-96)
    for i, idx in enumerate(range(87, 97)):
        angle = 2 * np.pi * i / 10
        lm[idx] = [cx + 30 + 15 * np.cos(angle), cy - 40 + 8 * np.sin(angle)]

    # Additional (97-105)
    for i, idx in enumerate(range(97, 106)):
        lm[idx] = [cx - 52 + i * 8, cy - 85]

    face.landmark_2d_106 = lm
    face.bbox = np.array([cx - 100, cy - 130, cx + 100, cy + 130], dtype=np.float32)
    face.kps = np.array(
        [[cx - 30, cy - 40], [cx + 30, cy - 40], [cx, cy - 10], [cx - 25, cy + 30], [cx + 25, cy + 30]],
        dtype=np.float32,
    )
    return face


# ---------------------------------------------------------------------------
# _eye_dimensions
# ---------------------------------------------------------------------------
class TestEyeDimensions:
    def _get_fn(self):
        from modules.processors.frame.face_masking import _eye_dimensions

        return _eye_dimensions

    def test_basic_rectangle(self):
        fn = self._get_fn()
        points = np.array([[10, 20], [50, 20], [50, 40], [10, 40]], dtype=np.float32)
        w, h = fn(points, scale=1.0)
        assert w == 40
        assert h == 20

    def test_scale_doubles(self):
        fn = self._get_fn()
        points = np.array([[10, 20], [50, 20], [50, 40], [10, 40]], dtype=np.float32)
        w, h = fn(points, scale=2.0)
        assert w == 80
        assert h == 40

    def test_single_point(self):
        fn = self._get_fn()
        points = np.array([[100, 200]], dtype=np.float32)
        w, h = fn(points, scale=1.0)
        assert w == 0
        assert h == 0


# ---------------------------------------------------------------------------
# _ellipse_polygon
# ---------------------------------------------------------------------------
class TestEllipsePolygon:
    def _get_fn(self):
        from modules.processors.frame.face_masking import _ellipse_polygon

        return _ellipse_polygon

    def test_returns_correct_number_of_points(self):
        fn = self._get_fn()
        pts = fn((100, 100), (30, 20), n=64)
        assert pts.shape == (64, 2)

    def test_dtype_is_int32(self):
        fn = self._get_fn()
        pts = fn((100, 100), (30, 20))
        assert pts.dtype == np.int32

    def test_center_is_approximate_centroid(self):
        fn = self._get_fn()
        center = (200, 150)
        pts = fn(center, (40, 30), n=128)
        mean = pts.mean(axis=0)
        assert abs(mean[0] - center[0]) < 2
        assert abs(mean[1] - center[1]) < 2

    def test_axes_bound_points(self):
        fn = self._get_fn()
        cx, cy = 100, 100
        ax, ay = 30, 20
        pts = fn((cx, cy), (ax, ay), n=128)
        assert np.all(pts[:, 0] >= cx - ax - 1)
        assert np.all(pts[:, 0] <= cx + ax + 1)
        assert np.all(pts[:, 1] >= cy - ay - 1)
        assert np.all(pts[:, 1] <= cy + ay + 1)


# ---------------------------------------------------------------------------
# _curved_eyebrow_contour
# ---------------------------------------------------------------------------
class TestCurvedEyebrowContour:
    def _get_fn(self):
        from modules.processors.frame.face_masking import _curved_eyebrow_contour

        return _curved_eyebrow_contour

    def test_fewer_than_5_points_returns_input(self):
        fn = self._get_fn()
        pts = np.array([[10, 20], [30, 25], [50, 22]], dtype=np.float32)
        result = fn(pts)
        np.testing.assert_array_equal(result, pts)

    def test_5_points_produces_contour(self):
        fn = self._get_fn()
        pts = np.array([[10, 30], [20, 25], [30, 20], [40, 25], [50, 30]], dtype=np.float32)
        result = fn(pts)
        assert result.shape[0] > 5  # More points than input (interpolated)
        assert result.shape[1] == 2

    def test_output_does_not_mutate_input(self):
        fn = self._get_fn()
        pts = np.array([[10, 30], [20, 25], [30, 20], [40, 25], [50, 30]], dtype=np.float32)
        pts_copy = pts.copy()
        fn(pts)
        np.testing.assert_array_equal(pts, pts_copy)


# ---------------------------------------------------------------------------
# create_lower_mouth_mask
# ---------------------------------------------------------------------------
class TestCreateLowerMouthMask:
    def _get_fn(self):
        from modules.processors.frame.face_masking import create_lower_mouth_mask

        return create_lower_mouth_mask

    def _setup_globals(self):
        import modules.globals

        modules.globals.mask_down_size = 0.1
        modules.globals.mouth_mask_size = 1.0
        modules.globals.mask_blur_kernel = 15

    def test_none_face_returns_zero_mask(self):
        fn = self._get_fn()
        self._setup_globals()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask, cutout, box, poly = fn(None, frame)
        assert mask.shape == (480, 640)
        assert np.all(mask == 0)
        assert cutout is None
        assert box == (0, 0, 0, 0)

    def test_missing_landmarks_returns_zero_mask(self):
        fn = self._get_fn()
        self._setup_globals()
        face = SimpleNamespace()  # No landmark_2d_106
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask, cutout, box, poly = fn(face, frame)
        assert np.all(mask == 0)

    def test_too_few_landmarks_returns_zero_mask(self):
        fn = self._get_fn()
        self._setup_globals()
        face = SimpleNamespace()
        face.landmark_2d_106 = np.zeros((50, 2), dtype=np.float32)  # Only 50
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask, cutout, box, poly = fn(face, frame)
        assert np.all(mask == 0)

    def test_valid_face_returns_nonzero_mask(self):
        fn = self._get_fn()
        self._setup_globals()
        face = _make_face_with_landmarks()
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        mask, cutout, box, poly = fn(face, frame)
        assert mask.max() > 0
        assert cutout is not None
        assert box != (0, 0, 0, 0)
        assert poly is not None

    def test_mouth_box_within_frame_bounds(self):
        fn = self._get_fn()
        self._setup_globals()
        face = _make_face_with_landmarks()
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        mask, cutout, box, poly = fn(face, frame)
        min_x, min_y, max_x, max_y = box
        assert min_x >= 0
        assert min_y >= 0
        assert max_x <= 640
        assert max_y <= 480

    def test_nan_landmarks_returns_zero_mask(self):
        fn = self._get_fn()
        self._setup_globals()
        face = _make_face_with_landmarks()
        face.landmark_2d_106[55] = [np.nan, np.nan]  # Corrupt a lower lip point
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask, cutout, box, poly = fn(face, frame)
        assert np.all(mask == 0)


# ---------------------------------------------------------------------------
# apply_mask_area boundary conditions
# ---------------------------------------------------------------------------
class TestApplyMaskArea:
    def _get_fn(self):
        from modules.processors.frame.face_masking import apply_mask_area

        return apply_mask_area

    def _setup_globals(self):
        import modules.globals

        modules.globals.MOUTH_FEATHER_RADIUS = 10
        modules.globals.mask_feather_ratio = 12

    def test_none_cutout_returns_frame(self):
        fn = self._get_fn()
        self._setup_globals()
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = fn(
            frame,
            None,
            (10, 10, 50, 50),
            np.zeros((200, 200), dtype=np.uint8),
            np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.int32),
        )
        np.testing.assert_array_equal(result, frame)

    def test_none_polygon_returns_frame(self):
        fn = self._get_fn()
        self._setup_globals()
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        cutout = np.zeros((40, 40, 3), dtype=np.uint8)
        result = fn(frame, cutout, (10, 10, 50, 50), np.zeros((200, 200), dtype=np.uint8), None)
        np.testing.assert_array_equal(result, frame)

    def test_zero_area_box_returns_frame(self):
        fn = self._get_fn()
        self._setup_globals()
        frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        cutout = np.zeros((0, 0, 3), dtype=np.uint8)
        face_mask = np.zeros((200, 200), dtype=np.uint8)
        polygon = np.array([[50, 50], [50, 50], [50, 50]], dtype=np.int32)
        original = frame.copy()
        result = fn(frame, cutout, (50, 50, 50, 50), face_mask, polygon)
        np.testing.assert_array_equal(result, original)

    def test_valid_inputs_modifies_frame(self):
        fn = self._get_fn()
        self._setup_globals()
        frame = np.full((200, 200, 3), 100, dtype=np.uint8)
        cutout = np.full((40, 60, 3), 200, dtype=np.uint8)
        box = (50, 80, 110, 120)
        face_mask = np.zeros((200, 200), dtype=np.uint8)
        face_mask[70:130, 40:120] = 255
        polygon = np.array([[55, 85], [105, 85], [105, 115], [55, 115]], dtype=np.int32)

        result = fn(frame, cutout, box, face_mask, polygon)
        assert result.dtype == np.uint8
        assert result.shape == frame.shape
