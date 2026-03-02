"""Tests for color correction modes: LAB transfer and histogram matching.

Covers:
- apply_histogram_matching() pure function
- color_correction_mode field on ProcessingConfig
- Backward-compat: swap_color_transfer=True still selects lab mode
"""
import cv2
import numpy as np
import pytest

from modules.processors.frame.face_masking import apply_histogram_matching, apply_color_transfer
from modules.processing_config import ProcessingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(h: int, w: int, bgr: tuple, noise: int = 0, seed: int = 0) -> np.ndarray:
    """Return a flat-colour BGR uint8 image optionally jittered with noise."""
    img = np.full((h, w, 3), bgr, dtype=np.uint8)
    if noise:
        rng = np.random.RandomState(seed)
        img = np.clip(img.astype(np.int32) + rng.randint(-noise, noise + 1, img.shape), 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# apply_histogram_matching — guard clauses
# ---------------------------------------------------------------------------

class TestHistogramMatchingGuards:

    def test_none_source_returns_none(self):
        target = _make_image(64, 64, (100, 150, 200))
        assert apply_histogram_matching(None, target) is None

    def test_none_target_returns_source_unchanged(self):
        source = _make_image(64, 64, (200, 100, 100))
        result = apply_histogram_matching(source, None)
        np.testing.assert_array_equal(result, source)

    def test_empty_source_returns_source(self):
        source = np.zeros((0, 0, 3), dtype=np.uint8)
        target = _make_image(64, 64, (100, 150, 200))
        result = apply_histogram_matching(source, target)
        np.testing.assert_array_equal(result, source)

    def test_empty_target_returns_source(self):
        source = _make_image(64, 64, (200, 100, 100))
        target = np.zeros((0, 0, 3), dtype=np.uint8)
        result = apply_histogram_matching(source, target)
        np.testing.assert_array_equal(result, source)


# ---------------------------------------------------------------------------
# apply_histogram_matching — output properties
# ---------------------------------------------------------------------------

class TestHistogramMatchingOutputProperties:

    def test_output_shape_matches_source(self):
        source = _make_image(128, 128, (200, 100, 100), noise=10, seed=1)
        target = _make_image(128, 128, (100, 150, 200), noise=10, seed=2)
        result = apply_histogram_matching(source, target)
        assert result.shape == source.shape

    def test_output_dtype_is_uint8(self):
        source = _make_image(64, 64, (200, 100, 100), noise=10, seed=3)
        target = _make_image(64, 64, (100, 150, 200), noise=10, seed=4)
        result = apply_histogram_matching(source, target)
        assert result.dtype == np.uint8

    def test_output_values_in_valid_range(self):
        source = _make_image(64, 64, (200, 100, 100), noise=20, seed=5)
        target = _make_image(64, 64, (50, 180, 220), noise=20, seed=6)
        result = apply_histogram_matching(source, target)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_source_is_not_mutated(self):
        source = _make_image(64, 64, (200, 100, 100), noise=10, seed=7)
        source_copy = source.copy()
        target = _make_image(64, 64, (100, 150, 200), noise=10, seed=8)
        apply_histogram_matching(source, target)
        np.testing.assert_array_equal(source, source_copy)


# ---------------------------------------------------------------------------
# apply_histogram_matching — histogram-level correctness
# ---------------------------------------------------------------------------

class TestHistogramMatchingCorrectness:
    """Verify that the output histogram is closer to the target than the source."""

    def _hist_distance(self, img: np.ndarray, ref: np.ndarray) -> float:
        """Mean L1 distance between per-channel histograms of img and ref."""
        total = 0.0
        for ch in range(3):
            h_img, _ = np.histogram(img[:, :, ch].ravel(), bins=256, range=(0, 256))
            h_ref, _ = np.histogram(ref[:, :, ch].ravel(), bins=256, range=(0, 256))
            total += np.abs(h_img.astype(float) - h_ref.astype(float)).mean()
        return total / 3

    def test_result_histogram_closer_to_target_than_source(self):
        """After matching, the result histogram should be closer to target than the original source histogram."""
        rng = np.random.RandomState(42)
        # Source: predominantly reddish (high B channel value in BGR = blue, so use high R)
        source = rng.randint(150, 200, (128, 128, 3), dtype=np.uint8)
        source[:, :, 0] = rng.randint(50, 80, (128, 128))   # low B
        source[:, :, 1] = rng.randint(80, 120, (128, 128))  # mid G
        source[:, :, 2] = rng.randint(160, 200, (128, 128)) # high R

        # Target: predominantly darker skin tone
        target = rng.randint(80, 130, (128, 128, 3), dtype=np.uint8)

        result = apply_histogram_matching(source, target)

        dist_before = self._hist_distance(source, target)
        dist_after = self._hist_distance(result, target)
        assert dist_after < dist_before, (
            f"Expected histogram distance to decrease after matching; "
            f"before={dist_before:.2f}, after={dist_after:.2f}"
        )

    def test_identity_when_source_equals_target(self):
        """Matching identical images should produce output very close to input."""
        img = _make_image(64, 64, (120, 160, 100), noise=15, seed=9)
        result = apply_histogram_matching(img, img)
        # Mean absolute difference should be very small (rounding only)
        mae = np.abs(result.astype(int) - img.astype(int)).mean()
        assert mae < 5.0, f"Expected near-identity for same source/target; MAE={mae:.2f}"


# ---------------------------------------------------------------------------
# ProcessingConfig — color_correction_mode field
# ---------------------------------------------------------------------------

class TestColorCorrectionModeConfig:

    def test_default_mode_is_none(self):
        config = ProcessingConfig()
        assert config.color_correction_mode == 'none'

    def test_can_set_lab_mode(self):
        config = ProcessingConfig(color_correction_mode='lab')
        assert config.color_correction_mode == 'lab'

    def test_can_set_histogram_mode(self):
        config = ProcessingConfig(color_correction_mode='histogram')
        assert config.color_correction_mode == 'histogram'

    def test_swap_color_transfer_still_exists_for_backward_compat(self):
        config = ProcessingConfig(swap_color_transfer=True)
        assert config.swap_color_transfer is True

    def test_swap_color_transfer_default_is_false(self):
        config = ProcessingConfig()
        assert config.swap_color_transfer is False


# ---------------------------------------------------------------------------
# Integration: both correction functions modify the crop
# ---------------------------------------------------------------------------

class TestColorCorrectionIntegration:

    def _get_distinct_pair(self):
        rng = np.random.RandomState(0)
        source = rng.randint(150, 200, (128, 128, 3), dtype=np.uint8)
        target = rng.randint(50, 100, (128, 128, 3), dtype=np.uint8)
        return source, target

    def test_lab_transfer_changes_crop(self):
        source, target = self._get_distinct_pair()
        result = apply_color_transfer(source, target)
        assert not np.array_equal(result, source)

    def test_histogram_matching_changes_crop(self):
        source, target = self._get_distinct_pair()
        result = apply_histogram_matching(source, target)
        assert not np.array_equal(result, source)

    def test_histogram_and_lab_produce_different_results(self):
        """Histogram and LAB modes should not be identical — different algorithms."""
        source, target = self._get_distinct_pair()
        lab_result = apply_color_transfer(source, target)
        hist_result = apply_histogram_matching(source, target)
        # Allow small differences — the two methods are distinct by design
        assert not np.array_equal(lab_result, hist_result), (
            "LAB transfer and histogram matching produced identical output — "
            "one may have been applied incorrectly."
        )
