"""Tests for shared paste-back primitives (Phase 2B).

Verifies:
- inverse_affine_warp correctly inverts and warps
- create_feathered_mask produces correct mask shape and edge ramps
- create_feathered_mask_1c produces 1-channel equivalent
- blend_with_mask blends correctly with mask weights
"""
import cv2
import numpy as np
import pytest

from modules.paste_back import (
    inverse_affine_warp,
    create_feathered_mask,
    create_feathered_mask_1c,
    blend_with_mask,
)


class TestInverseAffineWarp:
    """Tests for inverse affine warp primitive."""

    def test_identity_warp_preserves_image(self):
        """Identity matrix should produce the same image (centered)."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        M = np.eye(2, 3, dtype=np.float64)
        result = inverse_affine_warp(img, M, (64, 64))
        # Identity inverse is also identity
        np.testing.assert_array_equal(result, img)

    def test_output_size_matches_target(self):
        img = np.full((128, 128, 3), 100, dtype=np.uint8)
        M = np.array([[1.0, 0, 50], [0, 1.0, 50]], dtype=np.float64)
        result = inverse_affine_warp(img, M, (640, 480))
        assert result.shape == (480, 640, 3)

    def test_works_with_float32_input(self):
        img = np.full((64, 64), 0.5, dtype=np.float32)
        M = np.eye(2, 3, dtype=np.float64)
        result = inverse_affine_warp(img, M, (64, 64), border_value=0.0)
        assert result.dtype == np.float32

    def test_translation_shifts_content(self):
        """A translation M should shift content when inverted."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[20:40, 20:40] = 255  # White square in center

        # M translates by (-50, -50), so inverse M shifts by (+50, +50)
        # placing the white square at (70-90, 70-90) in the output
        M = np.array([[1.0, 0, -50], [0, 1.0, -50]], dtype=np.float64)
        result = inverse_affine_warp(img, M, (200, 200), border_value=0)

        assert result.shape == (200, 200, 3)
        assert result.sum() > 0  # Content should be present at shifted position


class TestCreateFeatheredMask:
    """Tests for 3-channel feathered mask creation."""

    def test_output_shape(self):
        mask = create_feathered_mask(512)
        assert mask.shape == (512, 512, 3)

    def test_output_dtype(self):
        mask = create_feathered_mask(256)
        assert mask.dtype == np.float32

    def test_center_is_one(self):
        mask = create_feathered_mask(256, border_fraction=0.1)
        center = mask[128, 128]
        np.testing.assert_allclose(center, 1.0)

    def test_corners_are_zero(self):
        mask = create_feathered_mask(256, border_fraction=0.1)
        np.testing.assert_allclose(mask[0, 0], 0.0, atol=1e-6)

    def test_edges_ramp(self):
        """Mask should ramp from 0 to 1 at edges."""
        size = 256
        border_fraction = 0.1
        mask = create_feathered_mask(size, border_fraction)
        border = int(size * border_fraction)

        # Top edge should ramp up
        top_col = mask[:border, size // 2, 0]
        assert top_col[0] < top_col[-1]

    def test_range(self):
        mask = create_feathered_mask(128, border_fraction=0.05)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_matches_face_enhancer_cached_mask(self):
        """Should produce the same mask as face_enhancer._get_feathered_mask."""
        from modules.processors.frame.face_enhancer import _get_feathered_mask

        size = 512
        shared = create_feathered_mask(size, border_fraction=0.05)
        enhancer = _get_feathered_mask(size)
        np.testing.assert_array_equal(shared, enhancer)


class TestCreateFeatheredMask1c:
    """Tests for single-channel feathered mask."""

    def test_output_shape(self):
        mask = create_feathered_mask_1c(256)
        assert mask.shape == (256, 256)

    def test_output_dtype(self):
        mask = create_feathered_mask_1c(128)
        assert mask.dtype == np.float32

    def test_center_is_one(self):
        mask = create_feathered_mask_1c(256)
        assert mask[128, 128] == pytest.approx(1.0)

    def test_corners_near_zero(self):
        mask = create_feathered_mask_1c(256)
        assert mask[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_range(self):
        mask = create_feathered_mask_1c(128)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0


class TestBlendWithMask:
    """Tests for mask-based alpha blending."""

    def test_full_mask_returns_foreground(self):
        fg = np.full((64, 64, 3), 200, dtype=np.uint8)
        bg = np.full((64, 64, 3), 50, dtype=np.uint8)
        mask = np.ones((64, 64, 3), dtype=np.float32)

        result = blend_with_mask(fg, bg, mask)
        np.testing.assert_array_equal(result, fg)

    def test_zero_mask_returns_background(self):
        fg = np.full((64, 64, 3), 200, dtype=np.uint8)
        bg = np.full((64, 64, 3), 50, dtype=np.uint8)
        mask = np.zeros((64, 64, 3), dtype=np.float32)

        result = blend_with_mask(fg, bg, mask)
        np.testing.assert_array_equal(result, bg)

    def test_half_mask_blends(self):
        fg = np.full((64, 64, 3), 200, dtype=np.uint8)
        bg = np.full((64, 64, 3), 100, dtype=np.uint8)
        mask = np.full((64, 64, 3), 0.5, dtype=np.float32)

        result = blend_with_mask(fg, bg, mask)
        # 0.5 * 200 + 0.5 * 100 = 150
        np.testing.assert_array_equal(result, 150)

    def test_2d_mask(self):
        """Should accept a 2D (H, W) mask and broadcast to 3 channels."""
        fg = np.full((64, 64, 3), 200, dtype=np.uint8)
        bg = np.full((64, 64, 3), 100, dtype=np.uint8)
        mask = np.full((64, 64), 0.5, dtype=np.float32)

        result = blend_with_mask(fg, bg, mask)
        np.testing.assert_array_equal(result, 150)

    def test_output_dtype_is_uint8(self):
        fg = np.full((32, 32, 3), 200, dtype=np.uint8)
        bg = np.full((32, 32, 3), 100, dtype=np.uint8)
        mask = np.full((32, 32, 3), 0.5, dtype=np.float32)

        result = blend_with_mask(fg, bg, mask)
        assert result.dtype == np.uint8

    def test_handles_float32_inputs(self):
        fg = np.full((32, 32, 3), 200.0, dtype=np.float32)
        bg = np.full((32, 32, 3), 100.0, dtype=np.float32)
        mask = np.full((32, 32, 3), 0.5, dtype=np.float32)

        result = blend_with_mask(fg, bg, mask)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, 150)
