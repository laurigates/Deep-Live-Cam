"""Tests for histogram matching and color transfer mode dispatch."""
import numpy as np
import pytest

from modules.processors.frame.face_masking import (
    apply_color_transfer,
    apply_histogram_transfer,
    apply_color_transfer_mode,
    _match_channel_cdf,
)


# ---------------------------------------------------------------------------
# _match_channel_cdf
# ---------------------------------------------------------------------------
class TestMatchChannelCdf:
    """Verify the per-channel CDF matching helper."""

    def test_output_shape_matches_input(self):
        rng = np.random.RandomState(0)
        src = rng.randint(0, 256, (64, 64), dtype=np.uint8)
        tgt = rng.randint(0, 256, (64, 64), dtype=np.uint8)
        result = _match_channel_cdf(src, tgt)
        assert result.shape == src.shape

    def test_output_dtype_uint8(self):
        src = np.full((32, 32), 128, dtype=np.uint8)
        tgt = np.full((32, 32), 200, dtype=np.uint8)
        result = _match_channel_cdf(src, tgt)
        assert result.dtype == np.uint8

    def test_identical_input_returns_similar(self):
        """When source == target, output should be close to input."""
        img = np.full((32, 32), 100, dtype=np.uint8)
        result = _match_channel_cdf(img, img)
        np.testing.assert_array_equal(result, img)

    def test_output_range_0_255(self):
        rng = np.random.RandomState(1)
        src = rng.randint(0, 256, (64, 64), dtype=np.uint8)
        tgt = rng.randint(0, 256, (64, 64), dtype=np.uint8)
        result = _match_channel_cdf(src, tgt)
        assert result.min() >= 0
        assert result.max() <= 255


# ---------------------------------------------------------------------------
# apply_histogram_transfer
# ---------------------------------------------------------------------------
class TestApplyHistogramTransfer:
    """Verify the full histogram transfer function."""

    def test_returns_source_on_none_source(self):
        assert apply_histogram_transfer(None, np.zeros((10, 10, 3), np.uint8)) is None

    def test_returns_source_on_none_target(self):
        src = np.zeros((10, 10, 3), dtype=np.uint8)
        result = apply_histogram_transfer(src, None)
        np.testing.assert_array_equal(result, src)

    def test_returns_source_on_empty_source(self):
        src = np.empty((0, 0, 3), dtype=np.uint8)
        tgt = np.zeros((10, 10, 3), dtype=np.uint8)
        result = apply_histogram_transfer(src, tgt)
        assert result.size == 0

    def test_returns_source_on_empty_target(self):
        src = np.zeros((10, 10, 3), dtype=np.uint8)
        tgt = np.empty((0, 0, 3), dtype=np.uint8)
        result = apply_histogram_transfer(src, tgt)
        np.testing.assert_array_equal(result, src)

    def test_output_shape_and_dtype(self):
        rng = np.random.RandomState(42)
        src = rng.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        tgt = rng.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        result = apply_histogram_transfer(src, tgt)
        assert result.shape == src.shape
        assert result.dtype == np.uint8

    def test_output_pixel_range(self):
        rng = np.random.RandomState(42)
        src = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        tgt = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = apply_histogram_transfer(src, tgt)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_identical_input_invariant(self):
        """When source and target are the same image, output should be close."""
        img = np.full((64, 64, 3), [120, 130, 140], dtype=np.uint8)
        result = apply_histogram_transfer(img, img.copy())
        # Allow small rounding differences from LAB round-trip
        assert np.allclose(result.astype(float), img.astype(float), atol=2)

    def test_color_shift_with_different_tones(self):
        """Histogram transfer should shift colors towards target."""
        # Blue-ish source
        source = np.full((64, 64, 3), [200, 100, 100], dtype=np.uint8)
        # Warm target
        target = np.full((64, 64, 3), [100, 150, 200], dtype=np.uint8)
        result = apply_histogram_transfer(source, target)
        # Result should differ from source
        assert not np.array_equal(result, source)

    def test_histogram_closer_to_target(self):
        """Output histogram should be closer to target than source was."""
        rng = np.random.RandomState(42)
        source = rng.randint(50, 150, (128, 128, 3), dtype=np.uint8)
        target = rng.randint(150, 250, (128, 128, 3), dtype=np.uint8)
        result = apply_histogram_transfer(source, target)

        # Compare mean pixel values — result should be closer to target mean
        src_dist = abs(float(source.mean()) - float(target.mean()))
        res_dist = abs(float(result.mean()) - float(target.mean()))
        assert res_dist < src_dist


# ---------------------------------------------------------------------------
# apply_color_transfer_mode (dispatch)
# ---------------------------------------------------------------------------
class TestApplyColorTransferMode:
    """Verify dispatch function routes to correct implementation."""

    def _make_pair(self):
        rng = np.random.RandomState(42)
        src = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        tgt = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        return src, tgt

    def test_none_mode_returns_source(self):
        src, tgt = self._make_pair()
        result = apply_color_transfer_mode(src, tgt, "none")
        np.testing.assert_array_equal(result, src)

    def test_lab_mode_matches_apply_color_transfer(self):
        src, tgt = self._make_pair()
        expected = apply_color_transfer(src, tgt)
        result = apply_color_transfer_mode(src, tgt, "lab")
        np.testing.assert_array_equal(result, expected)

    def test_histogram_mode_matches_apply_histogram_transfer(self):
        src, tgt = self._make_pair()
        expected = apply_histogram_transfer(src, tgt)
        result = apply_color_transfer_mode(src, tgt, "histogram")
        np.testing.assert_array_equal(result, expected)

    def test_default_mode_is_none(self):
        src, tgt = self._make_pair()
        result = apply_color_transfer_mode(src, tgt)
        np.testing.assert_array_equal(result, src)

    def test_unknown_mode_returns_source(self):
        """Unknown modes should fall through to 'none' (return source)."""
        src, tgt = self._make_pair()
        result = apply_color_transfer_mode(src, tgt, "unknown")
        np.testing.assert_array_equal(result, src)


# ---------------------------------------------------------------------------
# ProcessingConfig.color_transfer_mode
# ---------------------------------------------------------------------------
class TestColorTransferModeConfig:
    """Verify the config field and validation."""

    def test_default_is_none_mode(self):
        from modules.processing_config import ProcessingConfig
        config = ProcessingConfig()
        assert config.color_transfer_mode == "none"

    def test_can_set_lab(self):
        from modules.processing_config import ProcessingConfig
        config = ProcessingConfig(color_transfer_mode="lab")
        assert config.color_transfer_mode == "lab"

    def test_can_set_histogram(self):
        from modules.processing_config import ProcessingConfig
        config = ProcessingConfig(color_transfer_mode="histogram")
        assert config.color_transfer_mode == "histogram"

    def test_invalid_mode_raises(self):
        from modules.processing_config import ProcessingConfig
        with pytest.raises(ValueError, match="color_transfer_mode"):
            ProcessingConfig(color_transfer_mode="invalid")
