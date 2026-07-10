"""Tests for the Live Cam / File Processing UI mode split (#75).

Covers _on_mode_change widget show/hide logic, the Live/Stop button toggle
(set_live_button_running), canonical mode persistence with validation, and
the ui_webcam session-state callbacks — all with tkinter/customtkinter
mocked via conftest.py.
"""

import inspect
import json
from unittest.mock import MagicMock, patch

import pytest

import modules.globals
import modules.ui as ui
import modules.ui_webcam as uw

# Module-level UI state touched by the mode-toggle logic.
_UI_STATE_ATTRS = [
    "_current_mode",
    "_target_frame_widget",
    "_swap_button_widget",
    "_start_button_widget",
    "_live_button_widget",
    "_settings_tabview_ref",
    "_live_controls_frame",
    "_live_start_command",
]

# modules.globals attributes mutated by load_switch_states().
_GLOBALS_ATTRS = [
    "keep_fps",
    "keep_audio",
    "keep_frames",
    "use_png_frames",
    "many_faces",
    "map_faces",
    "poisson_blend",
    "color_correction",
    "color_correction_mode",
    "nsfw_filter",
    "live_mirror",
    "live_resizable",
    "show_fps",
    "virtual_cam",
    "mouth_mask",
    "show_mouth_mask_box",
    "occlusion_mask",
    "rife_enabled",
    "rife_model",
    "rife_multiplier",
    "half_rate_processing",
    "keyframe_interval",
    "live_max_fps",
    "landmark_smoothing",
    "landmark_smoothing_alpha",
    "prepaste_upscale_max",
    "scale_smoothing",
    "scale_smoothing_alpha",
    "source_path",
    "target_path",
]


@pytest.fixture
def ui_state():
    """Snapshot/restore ui module state; silence save_switch_states."""
    snapshot = {attr: getattr(ui, attr) for attr in _UI_STATE_ATTRS}
    with patch.object(ui, "save_switch_states", MagicMock()):
        yield ui
    for attr, value in snapshot.items():
        setattr(ui, attr, value)


@pytest.fixture
def globals_snapshot():
    """Snapshot/restore modules.globals attrs mutated by load_switch_states."""
    snap = {attr: getattr(modules.globals, attr) for attr in _GLOBALS_ATTRS}
    fp_ui = dict(modules.globals.fp_ui)
    frame_processors = list(modules.globals.frame_processors)
    yield
    for attr, value in snap.items():
        setattr(modules.globals, attr, value)
    modules.globals.fp_ui = fp_ui
    modules.globals.frame_processors = frame_processors


def _install_mock_widgets():
    ui._target_frame_widget = MagicMock()
    ui._swap_button_widget = MagicMock()
    ui._start_button_widget = MagicMock()
    ui._live_button_widget = MagicMock()
    ui._settings_tabview_ref = MagicMock()
    ui._live_controls_frame = MagicMock()
    ui._live_start_command = MagicMock()


class TestModeConstants:
    def test_canonical_constants(self):
        assert ui.MODE_LIVE == "Live Cam"
        assert ui.MODE_FILE == "File Processing"

    def test_settings_tabview_no_longer_takes_live_button(self):
        sig = inspect.signature(ui._add_settings_tabview)
        assert "live_button" not in sig.parameters

    def test_camera_controls_helper_exists(self):
        sig = inspect.signature(ui._add_camera_controls)
        assert set(sig.parameters) == {"parent", "root", "live_button"}


class TestOnModeChange:
    def test_switch_to_file_processing(self, ui_state):
        _install_mock_widgets()
        with patch.object(ui, "stop_active_session") as stop_mock:
            ui._on_mode_change(ui.MODE_FILE)

        assert ui._current_mode == ui.MODE_FILE
        ui._target_frame_widget.grid.assert_called_once_with()
        ui._swap_button_widget.grid.assert_called_once_with()
        ui._start_button_widget.grid.assert_called_once_with()
        ui._live_button_widget.grid_remove.assert_called_once_with()
        ui._live_controls_frame.grid_remove.assert_called_once_with()
        ui._settings_tabview_ref.set.assert_called_once_with("Export")
        stop_mock.assert_called_once()
        ui.save_switch_states.assert_called_once()

    def test_switch_to_live_cam(self, ui_state):
        _install_mock_widgets()
        ui._current_mode = ui.MODE_FILE
        with patch.object(ui, "stop_active_session") as stop_mock:
            ui._on_mode_change(ui.MODE_LIVE)

        assert ui._current_mode == ui.MODE_LIVE
        ui._target_frame_widget.grid_remove.assert_called_once_with()
        ui._swap_button_widget.grid_remove.assert_called_once_with()
        ui._start_button_widget.grid_remove.assert_called_once_with()
        ui._live_button_widget.grid.assert_called_once_with()
        ui._live_controls_frame.grid.assert_called_once_with()
        ui._settings_tabview_ref.set.assert_called_once_with("Live Mode")
        stop_mock.assert_not_called()

    def test_file_mode_resets_live_button_to_start_state(self, ui_state):
        """Switching away from Live Cam ends the session and restores 'Live'."""
        _install_mock_widgets()
        start_cmd = MagicMock()
        ui._live_start_command = start_cmd
        with patch.object(ui, "stop_active_session"):
            ui._on_mode_change(ui.MODE_FILE)
        kwargs = ui._live_button_widget.configure.call_args.kwargs
        assert kwargs["text"] == "Live"
        assert kwargs["command"] is start_cmd

    def test_none_widgets_are_safe(self, ui_state):
        """Mode change with no widgets built yet must not raise."""
        for attr in _UI_STATE_ATTRS[1:]:
            setattr(ui, attr, None)
        with patch.object(ui, "stop_active_session"):
            ui._on_mode_change(ui.MODE_FILE)
            ui._on_mode_change(ui.MODE_LIVE)

    def test_translated_label_stored_canonically(self, ui_state):
        """The segmented button passes translated labels; persisted mode
        must stay canonical (untranslated) so a language switch cannot
        corrupt the saved state."""
        _install_mock_widgets()
        translations = {"Live Cam": "Kamera", "File Processing": "Tiedostot"}
        with patch.object(ui, "_", side_effect=lambda s: translations.get(s, s)):
            with patch.object(ui, "stop_active_session"):
                ui._on_mode_change("Tiedostot")
                assert ui._current_mode == ui.MODE_FILE
                ui._on_mode_change("Kamera")
                assert ui._current_mode == ui.MODE_LIVE


class TestLiveStopButton:
    def test_running_true_shows_stop(self, ui_state):
        btn = MagicMock()
        ui._live_button_widget = btn
        ui.set_live_button_running(True)
        kwargs = btn.configure.call_args.kwargs
        assert kwargs["text"] == "Stop"
        assert callable(kwargs["command"])

    def test_running_false_restores_live_and_start_command(self, ui_state):
        btn = MagicMock()
        start_cmd = MagicMock()
        ui._live_button_widget = btn
        ui._live_start_command = start_cmd
        ui.set_live_button_running(False)
        kwargs = btn.configure.call_args.kwargs
        assert kwargs["text"] == "Live"
        assert kwargs["command"] is start_cmd

    def test_noop_when_button_missing(self, ui_state):
        ui._live_button_widget = None
        ui.set_live_button_running(True)
        ui.set_live_button_running(False)  # must not raise

    def test_stop_command_stops_active_session(self, ui_state):
        with patch.object(ui, "stop_active_session") as stop_mock:
            ui._stop_live()
        stop_mock.assert_called_once()


class TestModePersistence:
    def _load_with_state(self, tmp_path, monkeypatch, state: dict):
        state_file = tmp_path / "switch_states.json"
        state_file.write_text(json.dumps(state))
        monkeypatch.setattr(ui, "_state_file_path", lambda: str(state_file))
        ui.load_switch_states()

    def test_load_restores_file_mode(self, ui_state, globals_snapshot, tmp_path, monkeypatch):
        self._load_with_state(tmp_path, monkeypatch, {"ui_mode": "File Processing"})
        assert ui._current_mode == ui.MODE_FILE

    def test_load_restores_live_mode(self, ui_state, globals_snapshot, tmp_path, monkeypatch):
        self._load_with_state(tmp_path, monkeypatch, {"ui_mode": "Live Cam"})
        assert ui._current_mode == ui.MODE_LIVE

    def test_load_rejects_unknown_mode(self, ui_state, globals_snapshot, tmp_path, monkeypatch):
        """A stale/translated persisted value must fall back to Live Cam,
        otherwise CTkSegmentedButton.set() raises at startup."""
        self._load_with_state(tmp_path, monkeypatch, {"ui_mode": "garbage"})
        assert ui._current_mode == ui.MODE_LIVE

    def test_load_defaults_to_live_mode(self, ui_state, globals_snapshot, tmp_path, monkeypatch):
        self._load_with_state(tmp_path, monkeypatch, {})
        assert ui._current_mode == ui.MODE_LIVE

    def test_save_persists_canonical_mode(self, globals_snapshot, tmp_path, monkeypatch):
        state_file = tmp_path / "switch_states.json"
        monkeypatch.setattr(ui, "_state_file_path", lambda: str(state_file))
        original_mode = ui._current_mode
        try:
            ui._current_mode = ui.MODE_FILE
            ui.save_switch_states()
        finally:
            ui._current_mode = original_mode
        saved = json.loads(state_file.read_text())
        assert saved["ui_mode"] == "File Processing"


class TestWebcamSessionButtonState:
    """create_webcam_preview marks the Live button running; cleanup resets it."""

    def setup_method(self):
        self._orig_stop_event = uw._active_stop_event
        uw._active_stop_event = None
        uw._session_ready.set()

    def teardown_method(self):
        if uw._active_stop_event is not None:
            uw._active_stop_event.set()
        uw._active_stop_event = self._orig_stop_event
        uw._session_ready.set()

    def _run_preview(self, cap_start_ok: bool):
        cap = MagicMock()
        cap.start.return_value = cap_start_ok
        cap.read.return_value = (False, None)  # capture thread exits immediately
        root = MagicMock()
        preview = MagicMock()

        with (
            patch.object(uw, "VideoCapturer", return_value=cap),
            patch.object(uw, "set_det_size"),
            patch.object(uw, "reset_scale_smoother"),
            patch.object(uw, "get_frame_processors_modules", return_value=[]),
            patch.object(uw, "virtual_cam", MagicMock()),
            patch.object(uw, "cleanup_rife"),
            patch.object(ui, "ROOT", root),
            patch.object(ui, "PREVIEW", preview),
            patch.object(ui, "_preview_embedded", True),
            patch.object(ui, "update_status"),
            patch.object(ui, "set_live_button_running") as button_mock,
        ):
            uw.create_webcam_preview(0)
            yield_state = {"root": root, "button_mock": button_mock}
            return yield_state

    def test_successful_start_marks_button_running(self):
        state = self._run_preview(cap_start_ok=True)
        state["button_mock"].assert_called_once_with(True)
        assert state["root"].after.called

    def test_failed_camera_does_not_mark_running(self):
        state = self._run_preview(cap_start_ok=False)
        state["button_mock"].assert_not_called()

    def test_cleanup_resets_button(self):
        cap = MagicMock()
        cap.start.return_value = True
        cap.read.return_value = (False, None)
        root = MagicMock()
        preview = MagicMock()

        with (
            patch.object(uw, "VideoCapturer", return_value=cap),
            patch.object(uw, "set_det_size"),
            patch.object(uw, "reset_scale_smoother"),
            patch.object(uw, "get_frame_processors_modules", return_value=[]),
            patch.object(uw, "virtual_cam", MagicMock()),
            patch.object(uw, "cleanup_rife"),
            patch.object(ui, "ROOT", root),
            patch.object(ui, "PREVIEW", preview),
            patch.object(ui, "_preview_embedded", True),
            patch.object(ui, "update_status"),
            patch.object(ui, "set_live_button_running") as button_mock,
        ):
            uw.create_webcam_preview(0)
            # Grab the display-loop callback scheduled via ROOT.after
            display_func = root.after.call_args.args[1]
            button_mock.reset_mock()
            # The capture mock returns ret=False, so the capture thread sets
            # the stop event; wait for it, then run one display step which
            # must trigger _cleanup and reset the button.
            assert uw._active_stop_event.wait(timeout=2.0)
            display_func()
            button_mock.assert_any_call(False)
            # Background cleanup releases the session slot
            assert uw._session_ready.wait(timeout=2.0)
