"""Unit tests for scale smoothing, prepaste_upscale_max, and related ProcessingConfig fields.

Covers:
- _smooth_affine_scale() EMA behaviour
- reset_scale_smoother() resets module-level EMA state
- _paste_scale_from_M() respects different max_k values
- ProcessingConfig.scale_smoothing, scale_smoothing_alpha, prepaste_upscale_max defaults
- build_config_from_globals() includes prepaste_upscale_max from modules.globals
"""

import numpy as np
import pytest

import modules.globals
import modules.processors.frame.face_swapper as _swapper_mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _affine(sx: float, sy: float, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    """Return a 2x3 affine matrix with diagonal scale (sx, sy) and translation (tx, ty)."""
    return np.array(
        [[sx, 0.0, tx], [0.0, sy, ty]],
        dtype=np.float64,
    )


def _scale_from_M(M: np.ndarray) -> float:
    """Extract the Frobenius-style scale from the first row of M (same formula as implementation)."""
    return float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))


@pytest.fixture(autouse=True)
def _reset_scale_ema():
    """Guarantee a clean EMA state before and after every test in this module."""
    _swapper_mod.reset_scale_smoother()
    yield
    _swapper_mod.reset_scale_smoother()


# ---------------------------------------------------------------------------
# _smooth_affine_scale — first call (EMA is None)
# ---------------------------------------------------------------------------


class TestSmoothAffineScaleFirstCall:
    """On the first call after reset, EMA is None → return M unchanged."""

    def test_first_call_returns_m_unchanged(self):
        fn = _swapper_mod._smooth_affine_scale
        M = _affine(0.5, 0.5, 10.0, 20.0)
        original = M.copy()
        result = fn(M, alpha=0.3)
        np.testing.assert_array_equal(result, original)

    def test_first_call_does_not_modify_input_in_place(self):
        fn = _swapper_mod._smooth_affine_scale
        M = _affine(0.4, 0.4, 5.0, 5.0)
        original = M.copy()
        fn(M, alpha=0.3)
        # The original M passed in should not be mutated
        np.testing.assert_array_equal(M, original)

    def test_first_call_translation_preserved(self):
        fn = _swapper_mod._smooth_affine_scale
        M = _affine(0.5, 0.5, 30.0, 40.0)
        result = fn(M, alpha=0.3)
        assert result[0, 2] == pytest.approx(30.0)
        assert result[1, 2] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# _smooth_affine_scale — subsequent calls (EMA smoothing)
# ---------------------------------------------------------------------------


class TestSmoothAffineScaleEma:
    """On the second call, the scale should be EMA-blended."""

    def test_second_call_smooths_scale(self):
        """After first call seeds EMA, second call with different scale should produce smoothed output."""
        fn = _swapper_mod._smooth_affine_scale
        alpha = 0.3

        # First call: seeds EMA to scale=1.0 (identity-like)
        M1 = _affine(1.0, 1.0)
        fn(M1, alpha=alpha)

        # Second call with scale=0.5
        M2 = _affine(0.5, 0.5)
        result = fn(M2, alpha=alpha)

        # EMA = alpha*0.5 + (1-alpha)*1.0
        expected_scale = alpha * 0.5 + (1.0 - alpha) * 1.0
        actual_scale = _scale_from_M(result)
        assert actual_scale == pytest.approx(expected_scale, rel=1e-4)

    def test_ema_alpha_one_is_no_smoothing(self):
        """alpha=1.0 means no history — output scale equals current input scale."""
        fn = _swapper_mod._smooth_affine_scale
        M1 = _affine(1.0, 1.0)
        fn(M1, alpha=1.0)

        M2 = _affine(0.4, 0.4)
        result = fn(M2, alpha=1.0)
        assert _scale_from_M(result) == pytest.approx(0.4, rel=1e-4)

    def test_ema_alpha_zero_is_full_history(self):
        """alpha=0.0 means output scale equals the EMA history (previous scale)."""
        fn = _swapper_mod._smooth_affine_scale
        M1 = _affine(1.0, 1.0)
        fn(M1, alpha=0.0)  # seeds EMA = 1.0

        M2 = _affine(0.4, 0.4)
        result = fn(M2, alpha=0.0)
        # smoothed = 0*0.4 + 1*1.0 = 1.0
        assert _scale_from_M(result) == pytest.approx(1.0, rel=1e-4)

    def test_multiple_calls_converge(self):
        """Repeatedly calling with same scale should converge EMA to that scale."""
        fn = _swapper_mod._smooth_affine_scale
        alpha = 0.3
        target_scale = 0.6

        # Seed with a different scale
        fn(_affine(1.0, 1.0), alpha=alpha, input_size=128)

        # Drive many frames at target_scale
        M = _affine(target_scale, target_scale)
        last = None
        for _ in range(60):
            last = fn(M.copy(), alpha=alpha, input_size=128)

        assert _scale_from_M(last) == pytest.approx(target_scale, abs=0.01)


# ---------------------------------------------------------------------------
# _smooth_affine_scale — translation adjusted to preserve face center
# ---------------------------------------------------------------------------


class TestSmoothAffineScaleTranslation:
    """Translation column M[:, 2] must be adjusted so the face center stays anchored."""

    def test_translation_adjusted_on_second_call(self):
        """After smoothing changes scale, translation must follow t' = half*(1 - ratio) + ratio*t."""
        fn = _swapper_mod._smooth_affine_scale
        alpha = 0.3
        input_size = 128
        half = input_size / 2.0

        # Seed EMA with scale=1.0
        fn(_affine(1.0, 1.0), alpha=alpha, input_size=input_size)

        tx, ty = 99.0, 77.0
        M = _affine(0.5, 0.5, tx, ty)
        result = fn(M, alpha=alpha, input_size=input_size)

        # EMA = 0.3*0.5 + 0.7*1.0 = 0.85; ratio = 0.85/0.5 = 1.7
        smoothed = alpha * 0.5 + (1.0 - alpha) * 1.0
        ratio = smoothed / 0.5
        expected_tx = half * (1.0 - ratio) + ratio * tx
        expected_ty = half * (1.0 - ratio) + ratio * ty
        assert result[0, 2] == pytest.approx(expected_tx, rel=1e-6)
        assert result[1, 2] == pytest.approx(expected_ty, rel=1e-6)

    def test_translation_adjusted_with_256_input_size(self):
        """Verify translation formula works for Ghost/HyperSwap 256-sized models."""
        fn = _swapper_mod._smooth_affine_scale
        alpha = 0.3
        input_size = 256
        half = input_size / 2.0

        fn(_affine(1.0, 1.0), alpha=alpha, input_size=input_size)

        tx, ty = 50.0, 60.0
        M = _affine(0.5, 0.5, tx, ty)
        result = fn(M, alpha=alpha, input_size=input_size)

        smoothed = alpha * 0.5 + (1.0 - alpha) * 1.0
        ratio = smoothed / 0.5
        expected_tx = half * (1.0 - ratio) + ratio * tx
        expected_ty = half * (1.0 - ratio) + ratio * ty
        assert result[0, 2] == pytest.approx(expected_tx, rel=1e-6)
        assert result[1, 2] == pytest.approx(expected_ty, rel=1e-6)

    def test_translation_unchanged_when_scale_identical(self):
        """Even when smoothed ≈ current (no rescaling needed), translation must be intact."""
        fn = _swapper_mod._smooth_affine_scale
        # Seed with same scale so delta is ~0 and early-return path triggers
        fn(_affine(0.5, 0.5), alpha=0.3)
        # Second call: very close scale to trigger abs(smoothed-scale) < 1e-6 early return
        M = _affine(0.5, 0.5, 55.0, 66.0)
        result = fn(M, alpha=1.0)  # alpha=1 → smoothed == scale exactly
        assert result[0, 2] == pytest.approx(55.0)
        assert result[1, 2] == pytest.approx(66.0)


# ---------------------------------------------------------------------------
# _smooth_affine_scale — zero / degenerate scale guard
# ---------------------------------------------------------------------------


class TestSmoothAffineScaleDegenerate:
    def test_zero_scale_returns_m_unchanged(self):
        """M with zero scale should be returned as-is (guard against divide-by-zero)."""
        fn = _swapper_mod._smooth_affine_scale
        M = np.zeros((2, 3), dtype=np.float64)
        original = M.copy()
        result = fn(M, alpha=0.3)
        np.testing.assert_array_equal(result, original)


# ---------------------------------------------------------------------------
# reset_scale_smoother
# ---------------------------------------------------------------------------


class TestResetScaleSmoother:
    def test_reset_clears_ema_state(self):
        """After reset, the next call should behave as if it's the first call."""
        fn = _swapper_mod._smooth_affine_scale

        # Seed EMA
        fn(_affine(1.0, 1.0), alpha=0.3)

        # Reset
        _swapper_mod.reset_scale_smoother()

        # Post-reset: first call should return M unchanged (not smoothed)
        M = _affine(0.5, 0.5, 10.0, 20.0)
        original = M.copy()
        result = fn(M, alpha=0.3)
        np.testing.assert_array_equal(result, original)

    def test_reset_then_second_call_smoothes_normally(self):
        """After reset + two calls, EMA should be properly seeded again."""
        fn = _swapper_mod._smooth_affine_scale
        alpha = 0.3

        # Seed, reset, seed again
        fn(_affine(0.2, 0.2), alpha=alpha)
        _swapper_mod.reset_scale_smoother()

        fn(_affine(1.0, 1.0), alpha=alpha)  # re-seeds EMA = 1.0

        M2 = _affine(0.5, 0.5)
        result = fn(M2, alpha=alpha)
        expected_scale = alpha * 0.5 + (1.0 - alpha) * 1.0
        assert _scale_from_M(result) == pytest.approx(expected_scale, rel=1e-4)

    def test_reset_is_idempotent(self):
        """Calling reset when EMA is already None should not raise."""
        _swapper_mod.reset_scale_smoother()
        _swapper_mod.reset_scale_smoother()  # second reset — should not raise


# ---------------------------------------------------------------------------
# _paste_scale_from_M — respects max_k
# ---------------------------------------------------------------------------


class TestPasteScaleFromMMaxK:
    def _get_fn(self):
        from modules.processors.frame.face_swapper import _paste_scale_from_M

        return _paste_scale_from_M

    def test_default_max_k_is_4(self):
        """Small scale matrix should be capped at default max_k=4.0."""
        fn = self._get_fn()
        M = _affine(0.05, 0.05)  # k = 1/0.05 = 20 → capped to 4
        assert fn(M) == pytest.approx(4.0, abs=0.01)

    def test_custom_max_k_2(self):
        fn = self._get_fn()
        M = _affine(0.1, 0.1)  # k = 10 → capped to 2
        assert fn(M, max_k=2.0) == pytest.approx(2.0, abs=0.01)

    def test_custom_max_k_8(self):
        fn = self._get_fn()
        M = _affine(0.1, 0.1)  # k = 10 → capped to 8
        assert fn(M, max_k=8.0) == pytest.approx(8.0, abs=0.01)

    def test_scale_within_max_k_range(self):
        """When k < max_k, the uncapped value should be returned."""
        fn = self._get_fn()
        # scale = 0.5 → k = 2.0; max_k = 4.0 → should return 2.0 uncapped
        M = _affine(0.5, 0.5)
        assert fn(M, max_k=4.0) == pytest.approx(2.0, abs=0.01)

    def test_max_k_1_always_returns_1(self):
        """max_k=1 → always return 1.0 regardless of scale."""
        fn = self._get_fn()
        M = _affine(0.1, 0.1)
        assert fn(M, max_k=1.0) == pytest.approx(1.0, abs=0.01)

    def test_zero_scale_returns_1(self):
        fn = self._get_fn()
        M = np.zeros((2, 3), dtype=np.float64)
        assert fn(M, max_k=4.0) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# ProcessingConfig — new field defaults
# ---------------------------------------------------------------------------


class TestProcessingConfigNewFields:
    def test_scale_smoothing_default_false(self):
        from modules.processing_config import ProcessingConfig

        cfg = ProcessingConfig()
        assert cfg.scale_smoothing is False

    def test_scale_smoothing_alpha_default(self):
        from modules.processing_config import ProcessingConfig

        cfg = ProcessingConfig()
        assert cfg.scale_smoothing_alpha == pytest.approx(0.3)

    def test_prepaste_upscale_max_default(self):
        from modules.processing_config import ProcessingConfig

        cfg = ProcessingConfig()
        assert cfg.prepaste_upscale_max == pytest.approx(4.0)

    def test_scale_smoothing_can_be_set_true(self):
        from modules.processing_config import ProcessingConfig

        cfg = ProcessingConfig(scale_smoothing=True)
        assert cfg.scale_smoothing is True

    def test_scale_smoothing_alpha_custom(self):
        from modules.processing_config import ProcessingConfig

        cfg = ProcessingConfig(scale_smoothing_alpha=0.15)
        assert cfg.scale_smoothing_alpha == pytest.approx(0.15)

    def test_prepaste_upscale_max_custom(self):
        from modules.processing_config import ProcessingConfig

        cfg = ProcessingConfig(prepaste_upscale_max=6.0)
        assert cfg.prepaste_upscale_max == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# build_config_from_globals — includes prepaste_upscale_max and scale fields
# ---------------------------------------------------------------------------


class TestBuildConfigFromGlobalsScaleFields:
    @pytest.fixture(autouse=True)
    def _save_restore_globals(self):
        saved = {
            "prepaste_upscale_max": modules.globals.prepaste_upscale_max,
            "scale_smoothing": modules.globals.scale_smoothing,
            "scale_smoothing_alpha": modules.globals.scale_smoothing_alpha,
        }
        yield
        for key, val in saved.items():
            setattr(modules.globals, key, val)

    def test_prepaste_upscale_max_in_config(self):
        from modules.processing_config_factory import build_config_from_globals

        modules.globals.prepaste_upscale_max = 6.0
        cfg = build_config_from_globals()
        assert cfg.prepaste_upscale_max == pytest.approx(6.0)

    def test_prepaste_upscale_max_default_propagated(self):
        from modules.processing_config_factory import build_config_from_globals

        modules.globals.prepaste_upscale_max = 4.0
        cfg = build_config_from_globals()
        assert cfg.prepaste_upscale_max == pytest.approx(4.0)

    def test_scale_smoothing_propagated(self):
        from modules.processing_config_factory import build_config_from_globals

        modules.globals.scale_smoothing = True
        cfg = build_config_from_globals()
        assert cfg.scale_smoothing is True

    def test_scale_smoothing_alpha_propagated(self):
        from modules.processing_config_factory import build_config_from_globals

        modules.globals.scale_smoothing_alpha = 0.15
        cfg = build_config_from_globals()
        assert cfg.scale_smoothing_alpha == pytest.approx(0.15)
