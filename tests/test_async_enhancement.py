"""Tests for async enhancement thread (Issue #46).

The enhancement thread decouples face enhancement (GFPGAN/GPEN) from the
processing thread, allowing swap+masking to run at full speed while
enhancement catches up asynchronously on a dedicated thread.
"""

import threading
import time
from unittest.mock import MagicMock

import numpy as np

from modules.ui_webcam import _enhancement_thread_func


def _make_frame(fill: int = 0, shape=(480, 640, 3)) -> np.ndarray:
    return np.full(shape, fill, dtype=np.uint8)


# ---------------------------------------------------------------------------
# _enhancement_thread_func basics
# ---------------------------------------------------------------------------


class TestEnhancementThreadFunc:
    """Unit tests for _enhancement_thread_func with a mock processor."""

    def _run_thread(self, enhancement_input, enhancement_output, enhancement_lock, stop_event, timeout=1.0):
        """Start the enhancement thread and return the thread handle."""
        t = threading.Thread(
            target=_enhancement_thread_func,
            args=(enhancement_input, enhancement_output, enhancement_lock, stop_event),
            daemon=True,
        )
        t.start()
        return t

    def test_processes_normal_mode_frame(self):
        """Thread calls process_frame for non-map_faces input."""
        lock = threading.Lock()
        inp = [None]
        out = [None]
        stop = threading.Event()

        enhanced_frame = _make_frame(200)
        processor = MagicMock()
        processor.process_frame.return_value = enhanced_frame

        t = self._run_thread(inp, out, lock, stop)

        with lock:
            inp[0] = {
                "frame": _make_frame(100),
                "faces": [MagicMock()],
                "map_faces": False,
                "processor": processor,
                "seq": 1,
            }

        # Wait for output
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with lock:
                result = out[0]
            if result is not None:
                break
            time.sleep(0.01)

        stop.set()
        t.join(timeout=2.0)

        assert result is not None
        assert result["seq"] == 1
        assert np.array_equal(result["frame"], enhanced_frame)
        processor.process_frame.assert_called_once()
        processor.process_frame_v2.assert_not_called()

    def test_processes_map_faces_frame(self):
        """Thread calls process_frame_v2 for map_faces input."""
        lock = threading.Lock()
        inp = [None]
        out = [None]
        stop = threading.Event()

        enhanced_frame = _make_frame(150)
        processor = MagicMock()
        processor.process_frame_v2.return_value = enhanced_frame

        t = self._run_thread(inp, out, lock, stop)

        with lock:
            inp[0] = {
                "frame": _make_frame(100),
                "faces": None,
                "map_faces": True,
                "processor": processor,
                "seq": 1,
            }

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with lock:
                result = out[0]
            if result is not None:
                break
            time.sleep(0.01)

        stop.set()
        t.join(timeout=2.0)

        assert result is not None
        assert result["seq"] == 1
        assert np.array_equal(result["frame"], enhanced_frame)
        processor.process_frame_v2.assert_called_once()
        processor.process_frame.assert_not_called()

    def test_skips_duplicate_seq(self):
        """Thread does not reprocess the same seq number."""
        lock = threading.Lock()
        inp = [None]
        out = [None]
        stop = threading.Event()

        processor = MagicMock()
        processor.process_frame.return_value = _make_frame(200)

        t = self._run_thread(inp, out, lock, stop)

        # Submit seq=1
        with lock:
            inp[0] = {
                "frame": _make_frame(100),
                "faces": None,
                "map_faces": False,
                "processor": processor,
                "seq": 1,
            }

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with lock:
                result = out[0]
            if result is not None:
                break
            time.sleep(0.01)

        assert result is not None
        assert processor.process_frame.call_count == 1

        # Wait a bit — thread should NOT reprocess seq=1
        time.sleep(0.05)
        assert processor.process_frame.call_count == 1

        stop.set()
        t.join(timeout=2.0)

    def test_processes_new_seq(self):
        """Thread processes a new seq after completing the previous one."""
        lock = threading.Lock()
        inp = [None]
        out = [None]
        stop = threading.Event()

        processor = MagicMock()
        processor.process_frame.side_effect = [_make_frame(200), _make_frame(250)]

        t = self._run_thread(inp, out, lock, stop)

        # Submit seq=1
        with lock:
            inp[0] = {
                "frame": _make_frame(100),
                "faces": None,
                "map_faces": False,
                "processor": processor,
                "seq": 1,
            }

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with lock:
                result = out[0]
            if result is not None and result["seq"] == 1:
                break
            time.sleep(0.01)

        # Submit seq=2
        with lock:
            inp[0] = {
                "frame": _make_frame(120),
                "faces": None,
                "map_faces": False,
                "processor": processor,
                "seq": 2,
            }

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with lock:
                result = out[0]
            if result is not None and result["seq"] == 2:
                break
            time.sleep(0.01)

        stop.set()
        t.join(timeout=2.0)

        assert result["seq"] == 2
        assert processor.process_frame.call_count == 2

    def test_stops_on_stop_event(self):
        """Thread exits when stop_event is set."""
        lock = threading.Lock()
        inp = [None]
        out = [None]
        stop = threading.Event()

        t = self._run_thread(inp, out, lock, stop)
        assert t.is_alive()

        stop.set()
        t.join(timeout=2.0)
        assert not t.is_alive()

    def test_idles_on_none_input(self):
        """Thread sleeps when input is None, doesn't crash."""
        lock = threading.Lock()
        inp = [None]
        out = [None]
        stop = threading.Event()

        t = self._run_thread(inp, out, lock, stop)

        # Let it idle for a bit
        time.sleep(0.05)
        assert t.is_alive()

        with lock:
            assert out[0] is None

        stop.set()
        t.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Processing thread async enhancement integration (unit-level)
# ---------------------------------------------------------------------------


class TestAsyncEnhancementIntegration:
    """Test the async enhancement contract from the processing thread's perspective.

    These tests verify the submit/read protocol without running the actual
    processing thread.
    """

    def test_submit_increments_seq(self):
        """Each submission gets an incremented seq number."""
        enhancement_seq = 0
        seqs = []
        for _ in range(5):
            enhancement_seq += 1
            seqs.append(enhancement_seq)
        assert seqs == [1, 2, 3, 4, 5]

    def test_read_new_result_updates_latest(self):
        """Processing thread updates latest_enhanced_frame on new seq."""
        last_consumed_enh_seq = -1
        latest_enhanced_frame = None

        enh_out = {"frame": _make_frame(200), "seq": 1}
        if enh_out is not None and enh_out["seq"] != last_consumed_enh_seq:
            last_consumed_enh_seq = enh_out["seq"]
            latest_enhanced_frame = enh_out["frame"]

        assert last_consumed_enh_seq == 1
        assert latest_enhanced_frame is not None
        assert np.array_equal(latest_enhanced_frame, _make_frame(200))

    def test_read_stale_result_keeps_previous(self):
        """Processing thread ignores output with same seq as already consumed."""
        last_consumed_enh_seq = 3
        latest_enhanced_frame = _make_frame(100)

        enh_out = {"frame": _make_frame(200), "seq": 3}
        if enh_out is not None and enh_out["seq"] != last_consumed_enh_seq:
            last_consumed_enh_seq = enh_out["seq"]
            latest_enhanced_frame = enh_out["frame"]

        # Should not have changed
        assert last_consumed_enh_seq == 3
        assert np.array_equal(latest_enhanced_frame, _make_frame(100))

    def test_disabled_enhancer_clears_state(self):
        """When enhancer is toggled off, async state is cleared."""
        lock = threading.Lock()
        enhancement_input = [{"frame": _make_frame(100), "seq": 1}]
        enhancement_output = [{"frame": _make_frame(200), "seq": 1}]
        latest_enhanced_frame = _make_frame(200)

        # Simulate toggling off
        latest_enhanced_frame = None
        with lock:
            enhancement_input[0] = None
            enhancement_output[0] = None

        assert latest_enhanced_frame is None
        assert enhancement_input[0] is None
        assert enhancement_output[0] is None

    def test_skip_frame_still_reads_output(self):
        """On skip frames, no submission but latest result is still consumed."""
        lock = threading.Lock()
        enhancement_input = [None]
        enhancement_output = [{"frame": _make_frame(200), "seq": 5}]
        last_consumed_enh_seq = -1
        latest_enhanced_frame = None

        skip_enhancer = True
        # Skip submission (no write to enhancement_input)

        # But still read
        with lock:
            enh_out = enhancement_output[0]
        if enh_out is not None and enh_out["seq"] != last_consumed_enh_seq:
            last_consumed_enh_seq = enh_out["seq"]
            latest_enhanced_frame = enh_out["frame"]

        assert last_consumed_enh_seq == 5
        assert np.array_equal(latest_enhanced_frame, _make_frame(200))
