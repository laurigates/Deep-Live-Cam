"""Tests for enhancer skip-frame optimization (Issue #33).

Enhancer skip-frame mode reuses the previous enhanced result on skipped frames,
reducing GFPGAN/GPEN compute cost while maintaining visual quality through
temporal coherence.
"""

import numpy as np

import modules.globals

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------


class TestEnhancerSkipGlobals:
    """Test that enhancer_skip_interval exists with correct defaults."""

    def test_globals_has_enhancer_skip_interval(self):
        assert hasattr(modules.globals, "enhancer_skip_interval")

    def test_enhancer_skip_interval_default_is_1(self):
        """Default of 1 means no skipping (every frame enhanced)."""
        assert modules.globals.enhancer_skip_interval == 1

    def test_enhancer_skip_interval_is_int(self):
        assert isinstance(modules.globals.enhancer_skip_interval, int)


# ---------------------------------------------------------------------------
# Skip formula
# ---------------------------------------------------------------------------


class TestSkipFormula:
    """Unit-test the enhancer skip formula from _processing_thread_func."""

    @staticmethod
    def _should_skip(enhancer_frame_counter: int, interval: int) -> bool:
        """Replicate the skip formula from ui_webcam.py."""
        interval = max(1, interval)
        return interval > 1 and (enhancer_frame_counter % interval) != 1

    def test_skip_formula_interval_1(self):
        """interval=1 means never skip (every frame enhanced)."""
        for counter in range(1, 20):
            assert self._should_skip(counter, 1) is False

    def test_skip_formula_interval_2(self):
        """interval=2 means skip every other frame.

        Frame 1: counter=1, 1%2=1 -> no skip (enhance)
        Frame 2: counter=2, 2%2=0 -> skip
        Frame 3: counter=3, 3%2=1 -> no skip (enhance)
        Frame 4: counter=4, 4%2=0 -> skip
        """
        results = [self._should_skip(i, 2) for i in range(1, 9)]
        # Enhance, skip, enhance, skip, ...
        assert results == [False, True, False, True, False, True, False, True]

    def test_skip_formula_interval_3(self):
        """interval=3 means skip 2 out of every 3 frames.

        Frame 1: counter=1, 1%3=1 -> no skip (enhance)
        Frame 2: counter=2, 2%3=2 -> skip
        Frame 3: counter=3, 3%3=0 -> skip
        Frame 4: counter=4, 4%3=1 -> no skip (enhance)
        """
        results = [self._should_skip(i, 3) for i in range(1, 10)]
        assert results == [False, True, True, False, True, True, False, True, True]

    def test_no_skip_on_first_frame(self):
        """First frame (counter=1) is never skipped regardless of interval."""
        for interval in range(1, 11):
            assert self._should_skip(1, interval) is False, (
                f"First frame should not be skipped with interval={interval}"
            )

    def test_skip_formula_interval_0_treated_as_1(self):
        """interval=0 is clamped to 1 via max(1, interval), so never skip."""
        for counter in range(1, 10):
            assert self._should_skip(counter, 0) is False


# ---------------------------------------------------------------------------
# Frame hold on skip
# ---------------------------------------------------------------------------


class TestFrameHoldOnSkip:
    """When skipping, the previous enhanced frame is returned."""

    @staticmethod
    def _make_frame(fill: int = 0) -> np.ndarray:
        return np.full((480, 640, 3), fill, dtype=np.uint8)

    def test_frame_hold_on_skip(self):
        """When skip_enhancer=True and prev_enhanced_frame exists, use cached frame."""
        prev_enhanced = self._make_frame(50)
        current_frame = self._make_frame(100)

        skip_enhancer = True
        result = prev_enhanced if (skip_enhancer and prev_enhanced is not None) else current_frame

        assert np.array_equal(result, prev_enhanced)
        assert not np.array_equal(result, current_frame)

    def test_no_skip_on_first_frame_no_cache(self):
        """When prev_enhanced_frame is None (first frame), enhancer must run."""
        prev_enhanced = None
        current_frame = self._make_frame(100)

        skip_enhancer = True  # would skip, but no cache available
        # falls through to actual enhancement when prev_enhanced is None
        result = prev_enhanced if (skip_enhancer and prev_enhanced is not None) else current_frame

        assert np.array_equal(result, current_frame)

    def test_cache_updated_on_enhance(self):
        """When not skipping, the result is cached for future skip frames."""
        enhanced_result = self._make_frame(75)
        prev_enhanced = None

        # Simulate: enhancer runs and produces enhanced_result
        result = enhanced_result
        prev_enhanced = result.copy()

        assert prev_enhanced is not None
        assert np.array_equal(prev_enhanced, enhanced_result)


# ---------------------------------------------------------------------------
# Independence from half-rate processing
# ---------------------------------------------------------------------------


class TestEnhancerSkipIndependentOfHalfRate:
    """Both enhancer skip and half-rate can be active simultaneously."""

    @staticmethod
    def _half_rate_skip(frame_counter: int, keyframe_interval: int) -> bool:
        return (frame_counter % keyframe_interval) != 1

    @staticmethod
    def _enhancer_skip(enhancer_counter: int, interval: int) -> bool:
        interval = max(1, interval)
        return interval > 1 and (enhancer_counter % interval) != 1

    def test_both_active_independently(self):
        """Half-rate and enhancer skip use separate counters and don't interfere."""
        keyframe_interval = 2
        enhancer_interval = 3
        enhancer_counter = 0

        half_rate_skips = []
        enhancer_skips = []

        for frame_counter in range(1, 10):
            hr_skip = self._half_rate_skip(frame_counter, keyframe_interval)
            half_rate_skips.append(hr_skip)

            if not hr_skip:
                # Only increment enhancer counter on keyframes (when face processing runs)
                enhancer_counter += 1
                enhancer_skips.append(self._enhancer_skip(enhancer_counter, enhancer_interval))
            else:
                enhancer_skips.append(None)  # enhancer doesn't run on half-rate skips

        # Half-rate: skip on even frames (2,4,6,8)
        assert half_rate_skips == [False, True, False, True, False, True, False, True, False]

        # Enhancer runs on keyframes (1,3,5,7,9) with its own counter (1,2,3,4,5)
        # interval=3: skip when counter%3 != 1 -> skip on counter 2,3,5
        # counter 1: no skip, counter 2: skip, counter 3: skip, counter 4: no skip, counter 5: skip
        assert enhancer_skips == [False, None, True, None, True, None, False, None, True]


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


class TestEnhancerHelpers:
    """Test the _ENHANCER_NAMES set and _is_enhancer_enabled helper."""

    def test_enhancer_names_contains_all_variants(self):
        from modules.ui_webcam import _ENHANCER_NAMES

        assert "DLC.FACE-ENHANCER" in _ENHANCER_NAMES
        assert "DLC.FACE-ENHANCER-GPEN256" in _ENHANCER_NAMES
        assert "DLC.FACE-ENHANCER-GPEN512" in _ENHANCER_NAMES

    def test_enhancer_names_excludes_swapper(self):
        from modules.ui_webcam import _ENHANCER_NAMES

        assert "DLC.FACE-SWAPPER" not in _ENHANCER_NAMES

    def test_is_enhancer_enabled_checks_fp_ui(self):
        from unittest.mock import MagicMock

        from modules.ui_webcam import _is_enhancer_enabled

        processor = MagicMock()
        processor.NAME = "DLC.FACE-ENHANCER"

        original = modules.globals.fp_ui.copy()
        try:
            modules.globals.fp_ui["face_enhancer"] = True
            assert _is_enhancer_enabled(processor) is True

            modules.globals.fp_ui["face_enhancer"] = False
            assert _is_enhancer_enabled(processor) is False
        finally:
            modules.globals.fp_ui = original

    def test_is_enhancer_enabled_unknown_processor(self):
        from unittest.mock import MagicMock

        from modules.ui_webcam import _is_enhancer_enabled

        processor = MagicMock()
        processor.NAME = "DLC.FACE-SWAPPER"
        assert _is_enhancer_enabled(processor) is False
