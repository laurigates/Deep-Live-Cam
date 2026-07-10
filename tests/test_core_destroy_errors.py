"""Tests for error handling in core.destroy() — issue #104.

destroy() must never abort the shutdown sequence, but unexpected teardown
failures must be logged instead of silently swallowed.  Expected ImportErrors
(headless mode without the tkinter/UI stack) stay silent.
"""

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

import modules.globals


@pytest.fixture
def no_target_path():
    """Ensure clean_temp() is skipped and restore the global afterwards."""
    original = modules.globals.target_path
    modules.globals.target_path = None
    yield
    modules.globals.target_path = original


class TestDestroyErrorLogging:
    def test_webcam_teardown_failure_is_logged(self, caplog, no_target_path):
        """A raising stop_active_session must be logged, and shutdown must proceed."""
        import modules.core as core

        with (
            patch("modules.ui_webcam.stop_active_session", side_effect=RuntimeError("boom")),
            patch.dict(sys.modules, {"modules.ui": None}),  # UI import -> ImportError branch
            caplog.at_level(logging.ERROR, logger="modules.core"),
        ):
            with pytest.raises(SystemExit):
                core.destroy(to_quit=True)

        assert "Error stopping active webcam session" in caplog.text

    def test_ui_quit_failure_is_logged(self, caplog, no_target_path):
        """A raising ui.ROOT.quit() must be logged, and shutdown must proceed."""
        import modules as modules_pkg
        import modules.core as core

        fake_ui = MagicMock()
        fake_ui.ROOT.quit.side_effect = RuntimeError("tk exploded")

        with (
            patch("modules.ui_webcam.stop_active_session"),
            patch.dict(sys.modules, {"modules.ui": fake_ui}),
            # "import modules.ui as ui" binds via the parent package attribute
            # when the real module was imported earlier in the session, so the
            # sys.modules patch alone is not enough (bpo-30024 semantics).
            patch.object(modules_pkg, "ui", fake_ui, create=True),
            caplog.at_level(logging.ERROR, logger="modules.core"),
        ):
            with pytest.raises(SystemExit):
                core.destroy(to_quit=True)

        assert "Error quitting UI root during shutdown" in caplog.text

    def test_headless_import_errors_stay_silent(self, caplog, no_target_path):
        """ImportError (headless mode) must not produce error records."""
        import modules.core as core

        with (
            patch.dict(sys.modules, {"modules.ui_webcam": None, "modules.ui": None}),
            caplog.at_level(logging.ERROR, logger="modules.core"),
        ):
            with pytest.raises(SystemExit):
                core.destroy(to_quit=True)

        assert caplog.records == []

    def test_no_quit_skips_teardown(self, no_target_path):
        """destroy(to_quit=False) must not raise SystemExit."""
        import modules.core as core

        core.destroy(to_quit=False)  # must not raise
