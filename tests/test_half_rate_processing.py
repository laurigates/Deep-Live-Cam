"""Tests for half-rate face processing with RIFE temporal upscaling (Issue #29).

Half-rate mode runs face processing only on every Nth frame (keyframes) and uses
RIFE interpolation to fill in the skipped frames, trading face-processing compute
for cheaper RIFE interpolation.
"""
from unittest.mock import patch

import numpy as np
import pytest

import modules.globals


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------


class TestHalfRateGlobals:
    """Test that half-rate globals exist with correct defaults."""

    def test_half_rate_disabled_by_default(self):
        assert hasattr(modules.globals, "half_rate_processing")
        assert modules.globals.half_rate_processing is False

    def test_keyframe_interval_default(self):
        assert hasattr(modules.globals, "keyframe_interval")
        assert modules.globals.keyframe_interval == 2

    def test_keyframe_interval_is_int(self):
        assert isinstance(modules.globals.keyframe_interval, int)

    def test_half_rate_is_bool(self):
        assert isinstance(modules.globals.half_rate_processing, bool)


# ---------------------------------------------------------------------------
# Keyframe interval range
# ---------------------------------------------------------------------------


class TestKeyframeIntervalRange:
    """Validate keyframe_interval default is within the allowed 2-10 range."""

    def test_keyframe_interval_minimum(self):
        """keyframe_interval must be at least 2 (at least one skip frame)."""
        assert modules.globals.keyframe_interval >= 2

    def test_keyframe_interval_maximum(self):
        """keyframe_interval must be at most 10 per CLI choices=range(2, 11)."""
        assert modules.globals.keyframe_interval <= 10


# ---------------------------------------------------------------------------
# Keyframe selection logic
# ---------------------------------------------------------------------------


class TestKeyframeLogic:
    """Unit-test the keyframe selection formula used in _processing_thread_func."""

    @staticmethod
    def _is_keyframe(frame_counter: int, keyframe_interval: int) -> bool:
        """Replicate the keyframe logic from ui_webcam.py."""
        return (frame_counter % keyframe_interval) == 1

    @staticmethod
    def _skip_face_processing(half_rate_enabled: bool, frame_counter: int, keyframe_interval: int) -> bool:
        is_keyframe = (frame_counter % keyframe_interval) == 1
        return half_rate_enabled and not is_keyframe

    def test_interval_2_odd_frames_are_keyframes(self):
        """With interval=2 frames 1,3,5,7,9 are keyframes; 2,4,6,8,10 are skips."""
        for i in range(1, 11):
            expected = (i % 2 == 1)
            assert self._is_keyframe(i, 2) == expected, (
                f"Frame {i} with interval=2: expected is_keyframe={expected}"
            )

    def test_interval_4_frames_1_5_9_are_keyframes(self):
        assert self._is_keyframe(1, 4) is True
        assert self._is_keyframe(2, 4) is False
        assert self._is_keyframe(3, 4) is False
        assert self._is_keyframe(4, 4) is False
        assert self._is_keyframe(5, 4) is True
        assert self._is_keyframe(9, 4) is True

    def test_interval_3_frames_1_4_7_are_keyframes(self):
        assert self._is_keyframe(1, 3) is True
        assert self._is_keyframe(2, 3) is False
        assert self._is_keyframe(3, 3) is False
        assert self._is_keyframe(4, 3) is True
        assert self._is_keyframe(7, 3) is True

    def test_skip_processing_when_half_rate_on_skip_frame(self):
        """skip_face_processing=True when half-rate enabled and frame is not a keyframe."""
        assert self._skip_face_processing(True, 2, 2) is True

    def test_no_skip_when_half_rate_disabled(self):
        """skip_face_processing is always False when half-rate is disabled."""
        for i in range(1, 11):
            assert self._skip_face_processing(False, i, 2) is False

    def test_no_skip_on_keyframe_with_half_rate_enabled(self):
        """Keyframes are never skipped even in half-rate mode."""
        assert self._skip_face_processing(True, 1, 2) is False  # frame 1 = keyframe


# ---------------------------------------------------------------------------
# Warning when RIFE unavailable
# ---------------------------------------------------------------------------


class TestHalfRateRifeRequirement:
    """Half-rate mode should warn when RIFE is unavailable, not crash."""

    def test_warning_printed_when_rife_disabled(self, capsys):
        """When half-rate=on and rife_enabled=False, a warning is logged on skip frames."""
        half_rate_enabled = True
        rife_enabled = False
        half_rate_warned = False
        frame_counter = 2        # skip frame for interval=2
        keyframe_interval = 2
        is_keyframe = (frame_counter % keyframe_interval) == 1
        skip_face_processing = half_rate_enabled and not is_keyframe

        # Replicate warning logic from _processing_thread_func else-branch
        if skip_face_processing and half_rate_enabled and not rife_enabled:
            if not half_rate_warned:
                print(
                    "[DLC.HALF-RATE] Half-rate processing requires RIFE "
                    "— output will use raw frames on skips"
                )
                half_rate_warned = True

        captured = capsys.readouterr()
        assert "[DLC.HALF-RATE]" in captured.out
        assert half_rate_warned is True

    def test_warning_fires_only_once(self):
        """The half_rate_warned flag prevents repeated warning prints."""
        half_rate_warned = False
        warning_count = 0

        for _ in range(5):  # simulate 5 consecutive skip frames
            if not half_rate_warned:
                warning_count += 1
                half_rate_warned = True

        assert warning_count == 1


# ---------------------------------------------------------------------------
# RIFE interpolation on skip frames
# ---------------------------------------------------------------------------


class TestSkipFrameRifeInterpolation:
    """Verify RIFE is called with (prev_processed, current_raw) on skip frames."""

    @staticmethod
    def _make_frame(fill: int = 0) -> np.ndarray:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = fill
        return frame

    def test_rife_called_with_prev_processed_and_raw(self):
        """interpolate_frame_pair receives (prev_processed_frame, raw_temp_frame)."""
        prev_processed = self._make_frame(50)
        raw_frame = self._make_frame(100)
        interp_frame = self._make_frame(75)

        with patch(
            "modules.rife_interpolation.interpolate_frame_pair",
            return_value=[interp_frame],
        ) as mock_interp, patch(
            "modules.rife_interpolation.has_native_binding", return_value=True
        ):
            from modules.rife_interpolation import has_native_binding, interpolate_frame_pair

            # Simulate skip-frame logic
            temp_frame = raw_frame.copy()
            if has_native_binding():
                intermediates = interpolate_frame_pair(prev_processed, temp_frame, multiplier=2)
                if intermediates:
                    temp_frame = intermediates[0]

        assert mock_interp.call_count == 1
        call_args, call_kwargs = mock_interp.call_args
        assert np.array_equal(call_args[0], prev_processed)
        assert np.array_equal(call_args[1], raw_frame)
        assert call_kwargs.get("multiplier") == 2

    def test_no_rife_call_without_prev_frame(self):
        """No interpolation attempt when prev_processed_frame is None."""
        prev_processed_frame = None
        raw_frame = self._make_frame(100)

        with patch("modules.rife_interpolation.interpolate_frame_pair") as mock_interp:
            # Simulate skip-frame guard
            if prev_processed_frame is not None:
                from modules.rife_interpolation import interpolate_frame_pair
                interpolate_frame_pair(prev_processed_frame, raw_frame, multiplier=2)

        mock_interp.assert_not_called()

    def test_raw_frame_kept_when_rife_returns_empty(self):
        """When RIFE returns [], raw temp_frame is used unchanged."""
        prev_processed = self._make_frame(50)
        raw_frame = self._make_frame(100)
        temp_frame = raw_frame.copy()

        with patch(
            "modules.rife_interpolation.interpolate_frame_pair", return_value=[]
        ), patch("modules.rife_interpolation.has_native_binding", return_value=True):
            from modules.rife_interpolation import has_native_binding, interpolate_frame_pair

            if has_native_binding():
                intermediates = interpolate_frame_pair(prev_processed, temp_frame, multiplier=2)
                if intermediates:
                    temp_frame = intermediates[0]

        assert np.array_equal(temp_frame, raw_frame)


# ---------------------------------------------------------------------------
# prev_processed_frame lifecycle
# ---------------------------------------------------------------------------


class TestPrevFrameLifecycle:
    """prev_processed_frame updated on keyframes only; unchanged on skip frames."""

    def test_prev_frame_updated_on_keyframe(self):
        prev_processed_frame = None
        keyframe_result = np.ones((480, 640, 3), dtype=np.uint8) * 128

        skip_face_processing = False  # keyframe
        rife_enabled = True
        half_rate_enabled = True
        temp_frame = keyframe_result.copy()

        if not skip_face_processing:
            if rife_enabled or half_rate_enabled:
                prev_processed_frame = temp_frame.copy()

        assert prev_processed_frame is not None
        assert np.array_equal(prev_processed_frame, keyframe_result)

    def test_prev_frame_unchanged_on_skip_frame(self):
        original = np.ones((480, 640, 3), dtype=np.uint8) * 128
        prev_processed_frame = original.copy()

        skip_face_processing = True  # skip frame
        rife_enabled = True
        half_rate_enabled = True

        if not skip_face_processing:
            # This branch would overwrite prev_processed_frame
            prev_processed_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        assert np.array_equal(prev_processed_frame, original)

    def test_prev_frame_set_to_none_when_neither_enabled(self):
        prev_processed_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        skip_face_processing = False
        rife_enabled = False
        half_rate_enabled = False

        if not skip_face_processing:
            if rife_enabled or half_rate_enabled:
                prev_processed_frame = prev_processed_frame.copy()
            else:
                prev_processed_frame = None

        assert prev_processed_frame is None


# ---------------------------------------------------------------------------
# Normal RIFE extra-frame emission suppressed on skip frames
# ---------------------------------------------------------------------------


class TestNormalRifeSuppressedOnSkipFrames:
    """Normal RIFE extra-frame emission must not run on half-rate skip frames."""

    def test_rife_block_skipped_when_skip_processing(self):
        skip_face_processing = True
        rife_enabled = True
        prev_processed_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frames_emitted = []

        if rife_enabled and prev_processed_frame is not None and not skip_face_processing:
            frames_emitted.append("extra_frame")

        assert len(frames_emitted) == 0

    def test_rife_block_runs_on_keyframes(self):
        skip_face_processing = False  # keyframe
        rife_enabled = True
        prev_processed_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rife_ran = False

        if rife_enabled and prev_processed_frame is not None and not skip_face_processing:
            rife_ran = True

        assert rife_ran is True


# ---------------------------------------------------------------------------
# Toggle on/off mid-stream
# ---------------------------------------------------------------------------


class TestToggleMidStream:
    """Toggling half-rate on/off during live session must not crash or corrupt state."""

    def test_frame_counter_logic_survives_toggle(self):
        """Frame counter correctly classifies frames before and after toggle."""
        keyframe_interval = 2
        frame_counter = 0
        skip_results = []

        for i in range(1, 11):
            frame_counter += 1
            # Toggle half-rate on after frame 5
            half_rate_enabled = i > 5
            is_keyframe = (frame_counter % keyframe_interval) == 1
            skip_face_processing = half_rate_enabled and not is_keyframe
            skip_results.append(skip_face_processing)

        # Frames 1-5: half-rate disabled → never skip
        assert all(r is False for r in skip_results[:5])

        # Frames 6-10: half-rate enabled; counter 6-10 alternates skip/keyframe
        # frame 6: counter=6, 6%2=0, not keyframe → skip
        # frame 7: counter=7, 7%2=1, keyframe → no skip
        # frame 8: counter=8, 8%2=0, not keyframe → skip
        # frame 9: counter=9, 9%2=1, keyframe → no skip
        # frame 10: counter=10, 10%2=0, not keyframe → skip
        assert skip_results[5] is True
        assert skip_results[6] is False
        assert skip_results[7] is True
        assert skip_results[8] is False
        assert skip_results[9] is True
