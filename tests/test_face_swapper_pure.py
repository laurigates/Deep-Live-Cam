"""Unit tests for pure / near-pure functions in face_swapper.py.

These exercise _paste_back, _paste_scale_from_M, _upscale_crop_for_paste,
_clamp_opacity, and apply_post_processing without loading real models.
"""

import cv2
import numpy as np
import pytest

import modules.globals


@pytest.fixture(autouse=True)
def _reset_swapper_globals():
    """Ensure globals are pristine before/after each test."""
    saved = {
        "opacity": modules.globals.opacity,
        "sharpness": modules.globals.sharpness,
        "enable_interpolation": modules.globals.enable_interpolation,
    }
    yield
    for key, val in saved.items():
        setattr(modules.globals, key, val)


# ---------------------------------------------------------------------------
# _paste_scale_from_M
# ---------------------------------------------------------------------------
class TestPasteScaleFromM:
    """Tests for the affine-transform scale extraction."""

    def _get_fn(self):
        from modules.processors.frame.face_swapper import _paste_scale_from_M

        return _paste_scale_from_M

    def test_identity_returns_1(self):
        fn = self._get_fn()
        M = np.eye(2, 3, dtype=np.float64)
        assert fn(M) == pytest.approx(1.0, abs=0.01)

    def test_small_scale_returns_clamped_max(self):
        """When M maps a huge region into 128px, k should be capped at max_k."""
        fn = self._get_fn()
        # Scale of 0.1 means 1/0.1 = 10, but max_k=4 by default
        M = np.array([[0.1, 0, 0], [0, 0.1, 0]], dtype=np.float64)
        assert fn(M) == pytest.approx(4.0, abs=0.01)

    def test_custom_max_k(self):
        fn = self._get_fn()
        M = np.array([[0.1, 0, 0], [0, 0.1, 0]], dtype=np.float64)
        assert fn(M, max_k=2.0) == pytest.approx(2.0, abs=0.01)

    def test_zero_scale_returns_1(self):
        """Zero scale should not crash — returns 1.0."""
        fn = self._get_fn()
        M = np.zeros((2, 3), dtype=np.float64)
        assert fn(M) == pytest.approx(1.0, abs=0.01)

    def test_rotation_preserves_scale(self):
        """A 45-degree rotation without scaling should give k≈1."""
        fn = self._get_fn()
        angle = np.pi / 4
        M = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
            ],
            dtype=np.float64,
        )
        assert fn(M) == pytest.approx(1.0, abs=0.05)

    def test_half_scale(self):
        """M that halves the image: k = 1/0.5 = 2.0."""
        fn = self._get_fn()
        M = np.array([[0.5, 0, 0], [0, 0.5, 0]], dtype=np.float64)
        assert fn(M) == pytest.approx(2.0, abs=0.01)

    def test_larger_than_1_scale(self):
        """M that zooms in (scale > 1): k should be clamped to 1.0."""
        fn = self._get_fn()
        M = np.array([[2.0, 0, 0], [0, 2.0, 0]], dtype=np.float64)
        assert fn(M) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# _upscale_crop_for_paste
# ---------------------------------------------------------------------------
class TestUpscaleCropForPaste:
    """Tests for the pre-paste upscaling function."""

    def _get_fn(self):
        from modules.processors.frame.face_swapper import _upscale_crop_for_paste

        return _upscale_crop_for_paste

    def test_k_near_1_returns_unchanged(self):
        fn = self._get_fn()
        bgr = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        aimg = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        M = np.eye(2, 3, dtype=np.float64)

        bgr_out, aimg_out, M_out = fn(bgr, aimg, M, k=1.0)
        assert bgr_out is bgr  # Same object, not a copy
        assert aimg_out is aimg
        np.testing.assert_array_equal(M_out, M)

    def test_k_2_doubles_dimensions(self):
        fn = self._get_fn()
        bgr = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        aimg = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        M = np.array([[0.5, 0, 10], [0, 0.5, 20]], dtype=np.float64)

        bgr_out, aimg_out, M_out = fn(bgr, aimg, M, k=2.0)
        assert bgr_out.shape == (256, 256, 3)
        assert aimg_out.shape == (256, 256, 3)
        # M should be scaled by k=2
        np.testing.assert_allclose(M_out, M * 2.0, atol=1e-6)

    def test_k_fractional(self):
        fn = self._get_fn()
        bgr = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        aimg = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        M = np.eye(2, 3, dtype=np.float64)

        bgr_out, aimg_out, M_out = fn(bgr, aimg, M, k=1.5)
        assert bgr_out.shape[0] == 192
        assert bgr_out.shape[1] == 192

    def test_output_dtype_preserved(self):
        fn = self._get_fn()
        bgr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        aimg = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        M = np.eye(2, 3, dtype=np.float64)

        bgr_out, aimg_out, _ = fn(bgr, aimg, M, k=2.0)
        assert bgr_out.dtype == np.uint8
        assert aimg_out.dtype == np.uint8


# ---------------------------------------------------------------------------
# _paste_back
# ---------------------------------------------------------------------------
class TestPasteBack:
    """Tests for the face paste-back compositing function."""

    def _get_fn(self):
        from modules.processors.frame.face_swapper import _paste_back

        return _paste_back

    def test_output_shape_matches_target(self):
        fn = self._get_fn()
        target = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        bgr_fake = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        aimg = bgr_fake.copy()
        M = np.array([[1.0, 0, 256], [0, 1.0, 176]], dtype=np.float64)

        result = fn(bgr_fake, aimg, M, target)
        assert result.shape == target.shape
        assert result.dtype == np.uint8

    def test_identical_fake_and_aligned_returns_target(self):
        """When fake == aligned, the diff is zero → output ≈ target."""
        fn = self._get_fn()
        target = np.full((200, 200, 3), 100, dtype=np.uint8)
        crop = np.full((64, 64, 3), 100, dtype=np.uint8)
        M = np.array([[1.0, 0, 68], [0, 1.0, 68]], dtype=np.float64)

        result = fn(crop, crop.copy(), M, target)
        assert result.dtype == np.uint8

    def test_empty_mask_returns_target(self):
        """When M maps to outside the frame, mask is empty → returns target unchanged."""
        fn = self._get_fn()
        target = np.full((100, 100, 3), 50, dtype=np.uint8)
        crop = np.full((64, 64, 3), 200, dtype=np.uint8)
        # Translate far outside the frame
        M = np.array([[1.0, 0, 9999], [0, 1.0, 9999]], dtype=np.float64)

        result = fn(crop, crop.copy(), M, target)
        np.testing.assert_array_equal(result, target)


# ---------------------------------------------------------------------------
# _clamp_opacity
# ---------------------------------------------------------------------------
class TestClampOpacity:
    def _get_fn(self):
        from modules.processors.frame.face_swapper import _clamp_opacity

        return _clamp_opacity

    def test_normal_range(self):
        import modules.globals

        modules.globals.opacity = 0.5
        assert self._get_fn()() == pytest.approx(0.5)

    def test_clamps_above_1(self):
        import modules.globals

        modules.globals.opacity = 1.5
        assert self._get_fn()() == pytest.approx(1.0)

    def test_clamps_below_0(self):
        import modules.globals

        modules.globals.opacity = -0.3
        assert self._get_fn()() == pytest.approx(0.0)

    def test_missing_attr_returns_1(self):
        import modules.globals

        if hasattr(modules.globals, "opacity"):
            saved = modules.globals.opacity
            delattr(modules.globals, "opacity")
            try:
                assert self._get_fn()() == pytest.approx(1.0)
            finally:
                modules.globals.opacity = saved


# ---------------------------------------------------------------------------
# apply_post_processing
# ---------------------------------------------------------------------------
class TestApplyPostProcessing:
    """Tests for the sharpening + interpolation post-processor."""

    def _get_fn(self):
        from modules.processors.frame.face_swapper import apply_post_processing

        return apply_post_processing

    def test_no_bboxes_returns_frame_unchanged(self):
        fn = self._get_fn()
        import modules.globals

        modules.globals.sharpness = 0.5
        modules.globals.enable_interpolation = False

        frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        result = fn(frame, [])
        np.testing.assert_array_equal(result, frame)

    def test_sharpness_zero_skips_sharpening(self):
        fn = self._get_fn()
        import modules.globals

        modules.globals.sharpness = 0.0
        modules.globals.enable_interpolation = False

        frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        bbox = np.array([20, 20, 180, 180], dtype=np.float32)
        original = frame.copy()
        result = fn(frame, [bbox])
        np.testing.assert_array_equal(result, original)

    def test_sharpening_modifies_face_region(self):
        fn = self._get_fn()
        import modules.globals

        modules.globals.sharpness = 1.0
        modules.globals.enable_interpolation = False

        frame = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
        bbox = np.array([20, 20, 180, 180], dtype=np.float32)
        original = frame.copy()
        result = fn(frame, [bbox])

        # Sharpening should modify at least some pixels in the face region
        face_region_original = original[20:180, 20:180]
        face_region_result = result[20:180, 20:180]
        assert not np.array_equal(face_region_original, face_region_result)

    def test_invalid_bbox_skipped(self):
        fn = self._get_fn()
        import modules.globals

        modules.globals.sharpness = 1.0
        modules.globals.enable_interpolation = False

        frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        original = frame.copy()

        # Zero-area bbox
        bbox = np.array([50, 50, 50, 50], dtype=np.float32)
        result = fn(frame, [bbox])
        np.testing.assert_array_equal(result, original)

    def test_bbox_clamped_to_frame_bounds(self):
        fn = self._get_fn()
        import modules.globals

        modules.globals.sharpness = 0.5
        modules.globals.enable_interpolation = False

        frame = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        # bbox extends beyond frame
        bbox = np.array([-10, -10, 110, 110], dtype=np.float32)
        result = fn(frame, [bbox])
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_multiple_bboxes(self):
        fn = self._get_fn()
        import modules.globals

        modules.globals.sharpness = 0.5
        modules.globals.enable_interpolation = False

        frame = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
        bboxes = [
            np.array([10, 10, 90, 90], dtype=np.float32),
            np.array([110, 110, 190, 190], dtype=np.float32),
        ]
        result = fn(frame, bboxes)
        assert result.shape == frame.shape
