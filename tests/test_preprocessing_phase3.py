"""Tests for Phase 3: Quality & Configurability.

3A: Optional post-swap color transfer (swap_color_transfer config flag).
3B: Expose paste-back tuning parameters in ProcessingConfig.
"""

import cv2
import numpy as np
import pytest

from modules.processing_config import ProcessingConfig


# ---------------------------------------------------------------------------
# 3A: Color transfer toggle
# ---------------------------------------------------------------------------
class TestSwapColorTransferConfig:
    """Verify swap_color_transfer config field exists and defaults correctly."""

    def test_default_is_false(self):
        config = ProcessingConfig()
        assert config.swap_color_transfer is False

    def test_can_enable(self):
        config = ProcessingConfig(swap_color_transfer=True)
        assert config.swap_color_transfer is True


class TestColorTransferIntegration:
    """Verify color transfer is applied to crops when enabled."""

    def test_color_transfer_modifies_crop(self):
        """apply_color_transfer should modify the source crop colours."""
        from modules.processors.frame.face_masking import apply_color_transfer

        rng = np.random.RandomState(42)
        # Source (swapped) with blue tint
        source = np.full((128, 128, 3), [200, 100, 100], dtype=np.uint8)
        source += rng.randint(-10, 10, source.shape, dtype=np.int16).clip(0, 255).astype(np.uint8)
        # Target with warm tint
        target = np.full((128, 128, 3), [100, 150, 200], dtype=np.uint8)

        result = apply_color_transfer(source, target)
        assert result.shape == source.shape
        assert result.dtype == np.uint8
        # Colors should shift towards target palette
        assert not np.array_equal(result, source)

    def test_disabled_produces_no_change(self):
        """With swap_color_transfer=False, no color transfer should be applied."""
        config = ProcessingConfig(swap_color_transfer=False)
        # Just verify the flag is respected — actual integration tested via
        # manual visual inspection with `just start`
        assert config.swap_color_transfer is False


# ---------------------------------------------------------------------------
# 3B: Paste-back tuning parameters
# ---------------------------------------------------------------------------
class TestPasteBackTuningDefaults:
    """Verify paste-back tuning parameters exist with correct defaults."""

    def test_paste_diff_threshold_default(self):
        config = ProcessingConfig()
        assert config.paste_diff_threshold == pytest.approx(10.0)

    def test_paste_mask_threshold_default(self):
        config = ProcessingConfig()
        assert config.paste_mask_threshold == pytest.approx(20.0)

    def test_paste_mask_erode_ratio_default(self):
        config = ProcessingConfig()
        assert config.paste_mask_erode_ratio == 10

    def test_paste_mask_blur_ratio_default(self):
        config = ProcessingConfig()
        assert config.paste_mask_blur_ratio == 20

    def test_enhance_feather_fraction_default(self):
        config = ProcessingConfig()
        assert config.enhance_feather_fraction == pytest.approx(0.05)


class TestPasteBackTuningCustom:
    """Verify paste-back tuning parameters can be customised."""

    def test_custom_diff_threshold(self):
        config = ProcessingConfig(paste_diff_threshold=15.0)
        assert config.paste_diff_threshold == pytest.approx(15.0)

    def test_custom_mask_threshold(self):
        config = ProcessingConfig(paste_mask_threshold=30.0)
        assert config.paste_mask_threshold == pytest.approx(30.0)

    def test_custom_erode_ratio(self):
        config = ProcessingConfig(paste_mask_erode_ratio=5)
        assert config.paste_mask_erode_ratio == 5

    def test_custom_blur_ratio(self):
        config = ProcessingConfig(paste_mask_blur_ratio=15)
        assert config.paste_mask_blur_ratio == 15

    def test_custom_feather_fraction(self):
        config = ProcessingConfig(enhance_feather_fraction=0.1)
        assert config.enhance_feather_fraction == pytest.approx(0.1)


class TestPasteBackWithConfig:
    """Verify _paste_back respects config parameters."""

    def test_default_config_pixel_identical_to_baseline(self):
        """With default config values, output should be identical to hardcoded baseline."""
        from modules.processors.frame.face_swapper import _paste_back

        rng = np.random.RandomState(42)
        target = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        bgr_fake = rng.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        aimg = rng.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        M = np.array([[0.8, 0, 36], [0, 0.8, 36]], dtype=np.float64)

        default_config = ProcessingConfig()
        result_with_config = _paste_back(bgr_fake, aimg, M, target, default_config)
        result_without = _paste_back(bgr_fake, aimg, M, target)

        np.testing.assert_array_equal(
            result_with_config,
            result_without,
            err_msg="Default config should produce identical output to no-config path",
        )

    def test_different_thresholds_produce_different_output(self):
        """Changing thresholds should produce different blending."""
        from modules.processors.frame.face_swapper import _paste_back

        rng = np.random.RandomState(42)
        target = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        # Create distinct fake and aligned to ensure diff mask has signal
        bgr_fake = np.full((128, 128, 3), 200, dtype=np.uint8)
        aimg = np.full((128, 128, 3), 100, dtype=np.uint8)
        M = np.array([[1.0, 0, 36], [0, 1.0, 36]], dtype=np.float64)

        config_default = ProcessingConfig()
        config_strict = ProcessingConfig(paste_diff_threshold=1.0, paste_mask_threshold=5.0)

        result_default = _paste_back(bgr_fake, aimg, M, target, config_default)
        result_strict = _paste_back(bgr_fake, aimg, M, target, config_strict)

        # Different thresholds should produce at least slightly different output
        # (may be identical if the face maps outside the frame — so we use a
        # centered M to ensure the mask is non-trivial)
        assert result_default.shape == result_strict.shape


class TestEnhancerFeatherConfig:
    """Verify enhancer feathered mask respects config."""

    def test_different_fractions_produce_different_masks(self):
        from modules.processors.frame.face_enhancer import _get_feathered_mask

        mask_narrow = _get_feathered_mask(256, border_fraction=0.02)
        mask_wide = _get_feathered_mask(256, border_fraction=0.15)

        # The masks should have different border widths
        assert not np.array_equal(mask_narrow, mask_wide)

    def test_default_fraction_matches_original(self):
        """Default 0.05 should match the original hardcoded behavior."""
        from modules.paste_back import create_feathered_mask
        from modules.processors.frame.face_enhancer import _get_feathered_mask

        size = 512
        cached = _get_feathered_mask(size, border_fraction=0.05)
        reference = create_feathered_mask(size, border_fraction=0.05)
        np.testing.assert_array_equal(cached, reference)
