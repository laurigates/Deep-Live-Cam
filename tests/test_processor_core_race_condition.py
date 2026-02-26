"""Tests for modules/processors/frame/core.py — race condition on FRAME_PROCESSORS_MODULES."""
import threading
import time
from unittest.mock import patch, MagicMock
import pytest


def test_get_processors_snapshot_returns_copy():
    """Verify _get_processors_snapshot() returns a shallow copy under lock."""
    from modules.processors.frame import core

    # Reset and populate the list
    core.FRAME_PROCESSORS_MODULES = []
    mock_proc_1 = MagicMock()
    mock_proc_1.__name__ = "modules.processors.frame.face_swapper"
    core.FRAME_PROCESSORS_MODULES.append(mock_proc_1)

    # Get a snapshot
    snapshot = core._get_processors_snapshot()

    # Verify it's a copy, not the original list
    assert snapshot == core.FRAME_PROCESSORS_MODULES
    assert snapshot is not core.FRAME_PROCESSORS_MODULES
    assert len(snapshot) == 1


def test_set_frame_processors_thread_safe():
    """
    Verifies that set_frame_processors_modules_from_ui() is protected by a lock.

    This test ensures that mutations are atomic and don't interfere with
    concurrent reads.
    """
    from modules.processors.frame import core

    # Reset the global list
    core.FRAME_PROCESSORS_MODULES = []
    mock_proc_1 = MagicMock()
    mock_proc_1.__name__ = "modules.processors.frame.face_swapper"
    core.FRAME_PROCESSORS_MODULES.append(mock_proc_1)

    errors = []
    mutation_count = [0]

    def reader_thread():
        """Reader that uses the snapshot getter."""
        try:
            for _ in range(50):
                snapshot = core._get_processors_snapshot()
                # Iterate the snapshot (safe, immutable from this perspective)
                for proc in snapshot:
                    time.sleep(0.00001)
        except Exception as e:
            errors.append(f"Reader error: {e}")

    def writer_thread():
        """Writer that mutates the list via set_frame_processors_modules_from_ui."""
        try:
            with patch.object(core.modules.globals, 'fp_ui', {'face_swapper': False}):
                with patch('modules.processors.frame.core.load_frame_processor_module'):
                    for i in range(10):
                        core.set_frame_processors_modules_from_ui(['face_swapper'])
                        mutation_count[0] += 1
                        time.sleep(0.0001)
        except Exception as e:
            errors.append(f"Writer error: {e}")

    # Start multiple readers and a writer
    readers = [
        threading.Thread(target=reader_thread, daemon=True)
        for _ in range(3)
    ]
    writer = threading.Thread(target=writer_thread, daemon=True)

    for r in readers:
        r.start()
    writer.start()

    for r in readers:
        r.join(timeout=5)
    writer.join(timeout=5)

    # With proper locking, no errors should occur
    assert len(errors) == 0, f"Unexpected errors with lock in place: {errors}"
    assert mutation_count[0] > 0, "Writer thread should have executed mutations"


def test_get_frame_processors_modules_returns_snapshot():
    """
    Verify that get_frame_processors_modules() returns a snapshot,
    not a reference to the internal list.
    """
    from modules.processors.frame import core

    core.FRAME_PROCESSORS_MODULES = []
    mock_proc = MagicMock()
    mock_proc.__name__ = "modules.processors.frame.test_processor"
    core.FRAME_PROCESSORS_MODULES.append(mock_proc)

    # Mock the necessary globals
    with patch.object(core.modules.globals, 'fp_ui', {}):
        result = core.get_frame_processors_modules(['test_processor'])

        # Result should be a copy
        assert isinstance(result, list)
        assert len(result) > 0
        # Modifying the returned list should not affect the internal one
        original_len = len(core.FRAME_PROCESSORS_MODULES)
        result.clear()
        assert len(core.FRAME_PROCESSORS_MODULES) == original_len
