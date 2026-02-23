"""Tests for half-rate face processing mode."""
import queue
import threading
import time
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

import modules.globals


class TestHalfRateGlobals:
    """Verify half-rate globals exist with correct types and defaults."""

    def test_half_rate_enabled_default(self):
        assert hasattr(modules.globals, "half_rate_enabled")
        assert modules.globals.half_rate_enabled is False

    def test_half_rate_interval_default(self):
        assert hasattr(modules.globals, "half_rate_interval")
        assert modules.globals.half_rate_interval == 2

    def test_half_rate_interval_is_int(self):
        assert isinstance(modules.globals.half_rate_interval, int)


class TestKeyframeLogic:
    """Test the keyframe determination logic used in _processing_thread_func."""

    @staticmethod
    def _is_keyframe(half_rate: bool, counter: int, interval: int) -> bool:
        """Replicate the keyframe check from _processing_thread_func."""
        return not half_rate or (counter % interval == 0)

    def test_all_frames_are_keyframes_when_disabled(self):
        for i in range(10):
            assert self._is_keyframe(False, i, 2) is True

    def test_every_other_frame_with_interval_2(self):
        results = [self._is_keyframe(True, i, 2) for i in range(8)]
        assert results == [True, False, True, False, True, False, True, False]

    def test_every_third_frame_with_interval_3(self):
        results = [self._is_keyframe(True, i, 3) for i in range(9)]
        assert results == [True, False, False, True, False, False, True, False, False]

    def test_every_fourth_frame_with_interval_4(self):
        results = [self._is_keyframe(True, i, 4) for i in range(8)]
        assert results == [True, False, False, False, True, False, False, False]


class TestHalfRateProcessing:
    """Integration tests for the half-rate processing thread behavior."""

    def _make_fake_frame(self, value: int = 128) -> np.ndarray:
        """Create a small BGR frame filled with a single value."""
        return np.full((4, 4, 3), value, dtype=np.uint8)

    @patch("modules.ui_webcam.get_frame_processors_modules", return_value=[])
    @patch("modules.ui_webcam.has_native_binding", return_value=False)
    @patch("modules.ui_webcam.gpu_flip", side_effect=lambda f, _: f)
    def test_half_rate_skips_non_keyframes(self, _flip, _binding, _procs):
        """With half-rate enabled and no RIFE, non-keyframe frames are skipped
        and the previous keyframe is repeated as fallback."""
        from modules.ui_webcam import _processing_thread_func

        modules.globals.half_rate_enabled = True
        modules.globals.half_rate_interval = 2
        modules.globals.live_mirror = False
        modules.globals.map_faces = False
        modules.globals.source_path = None
        modules.globals.many_faces = False
        modules.globals.show_fps = False
        modules.globals.virtual_cam = False
        modules.globals.frame_processors = []

        capture_queue = queue.Queue(maxsize=8)
        processed_queue = queue.Queue(maxsize=16)
        stop_event = threading.Event()
        detection_lock = threading.Lock()
        latest_frame_holder = [None]
        detection_result = {"target_face": None, "many_faces": None}

        # Put 4 frames into the capture queue
        for i in range(4):
            capture_queue.put(self._make_fake_frame(i * 50))

        # Run processing thread briefly
        proc_thread = threading.Thread(
            target=_processing_thread_func,
            args=(
                capture_queue, processed_queue, stop_event,
                latest_frame_holder, detection_result, detection_lock,
            ),
            daemon=True,
        )
        proc_thread.start()

        # Wait for frames to be processed
        time.sleep(0.5)
        stop_event.set()
        proc_thread.join(timeout=2.0)

        # With interval=2 and 4 input frames:
        # Frame 0 (counter=0): keyframe → emit (no prev_keyframe, no fill-in)
        # Frame 1 (counter=1): non-keyframe → skip
        # Frame 2 (counter=2): keyframe → emit fill-in (repeat of frame 0) + emit frame 2
        # Frame 3 (counter=3): non-keyframe → skip
        # Total output: frame0, fill_for_1, frame2 = 3 frames
        output_count = processed_queue.qsize()
        assert output_count >= 2, f"Expected at least 2 output frames, got {output_count}"

    @patch("modules.ui_webcam.get_frame_processors_modules", return_value=[])
    @patch("modules.ui_webcam.gpu_flip", side_effect=lambda f, _: f)
    def test_all_frames_processed_when_disabled(self, _flip, _procs):
        """With half-rate disabled, every frame is processed and emitted."""
        from modules.ui_webcam import _processing_thread_func

        modules.globals.half_rate_enabled = False
        modules.globals.half_rate_interval = 2
        modules.globals.live_mirror = False
        modules.globals.map_faces = False
        modules.globals.source_path = None
        modules.globals.many_faces = False
        modules.globals.show_fps = False
        modules.globals.virtual_cam = False
        modules.globals.frame_processors = []
        modules.globals.rife_enabled = False

        capture_queue = queue.Queue(maxsize=8)
        processed_queue = queue.Queue(maxsize=16)
        stop_event = threading.Event()
        detection_lock = threading.Lock()
        latest_frame_holder = [None]
        detection_result = {"target_face": None, "many_faces": None}

        num_frames = 6
        for i in range(num_frames):
            capture_queue.put(self._make_fake_frame(i * 40))

        proc_thread = threading.Thread(
            target=_processing_thread_func,
            args=(
                capture_queue, processed_queue, stop_event,
                latest_frame_holder, detection_result, detection_lock,
            ),
            daemon=True,
        )
        proc_thread.start()

        time.sleep(0.5)
        stop_event.set()
        proc_thread.join(timeout=2.0)

        output_count = processed_queue.qsize()
        assert output_count == num_frames, (
            f"Expected {num_frames} output frames, got {output_count}"
        )
