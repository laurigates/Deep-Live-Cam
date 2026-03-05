"""Tests for stop_active_session() in ui_webcam."""

import threading

from modules.ui_webcam import stop_active_session


class TestStopActiveSession:
    """Verify stop_active_session signals the active session to stop."""

    def setup_method(self):
        """Reset module-level session state before each test."""
        import modules.ui_webcam as uw

        uw._active_stop_event = None
        uw._session_ready.set()

    def test_noop_when_no_session_running(self):
        """Calling stop when no session is active should be a safe no-op."""
        import modules.ui_webcam as uw

        assert uw._active_stop_event is None
        stop_active_session()
        # Should not raise, _active_stop_event stays None
        assert uw._active_stop_event is None

    def test_sets_stop_event_when_session_active(self):
        """An active session's stop event should be set."""
        import modules.ui_webcam as uw

        evt = threading.Event()
        uw._active_stop_event = evt
        assert not evt.is_set()
        stop_active_session()
        assert evt.is_set()

    def test_noop_when_stop_event_already_set(self):
        """Calling stop when the session was already stopping is a safe no-op."""
        import modules.ui_webcam as uw

        evt = threading.Event()
        evt.set()
        uw._active_stop_event = evt
        stop_active_session()
        assert evt.is_set()
