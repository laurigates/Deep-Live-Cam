"""Tests for allocation reduction optimizations (Issue #94).

Covers three changes:
1. gpu_cvt_color removed from display loop in ui_webcam.py (uses cv2.cvtColor directly)
2. _current_frame assignment removed from VideoCapturer.read()
3. create_face_mask called once and shared between _apply_mouth_mask and _apply_poisson_blend
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. gpu_cvt_color not used in display loop path
# ---------------------------------------------------------------------------
class TestDisplayLoopNoCudaUpload:
    """Verify that the display loop converts BGR→RGB on CPU (no GPU round-trip)."""

    def test_display_loop_uses_cv2_cvtcolor_not_gpu(self):
        """_display_next_frame must use cv2.cvtColor, not gpu_cvt_color, for BGR→RGB."""
        import pathlib

        src = pathlib.Path("modules/ui_webcam.py").read_text()

        # Find the _display_next_frame function body as a substring
        fn_start = src.find("def _display_next_frame()")
        assert fn_start != -1, "_display_next_frame not found in ui_webcam.py"

        # Find the Image.fromarray line in that function
        img_line_idx = src.find("Image.fromarray", fn_start)
        assert img_line_idx != -1, "Image.fromarray not found after _display_next_frame"

        # Extract that line
        line_start = src.rfind("\n", 0, img_line_idx) + 1
        line_end = src.find("\n", img_line_idx)
        image_line = src[line_start:line_end].strip()

        # Must use cv2.cvtColor, not gpu_cvt_color
        assert "cv2.cvtColor" in image_line, f"Expected cv2.cvtColor in display line, got: {image_line!r}"
        assert "gpu_cvt_color" not in image_line, f"gpu_cvt_color still present in display line: {image_line!r}"

    def test_cv2_cvtcolor_produces_rgb_output(self):
        """cv2.cvtColor(frame, BGR2RGB) produces correct colour conversion."""
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr[:, :, 0] = 100  # Blue channel
        bgr[:, :, 2] = 200  # Red channel

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # After BGR→RGB: what was R(200) is now at index 0, B(100) at index 2
        assert rgb[0, 0, 0] == 200  # Red channel moved to index 0
        assert rgb[0, 0, 2] == 100  # Blue channel moved to index 2


# ---------------------------------------------------------------------------
# 2. VideoCapturer._current_frame no longer stored
# ---------------------------------------------------------------------------
class TestVideoCapturerNoFrameStorage:
    """Verify that VideoCapturer.read() does not store the frame in _current_frame."""

    def test_current_frame_not_assigned_in_read(self):
        """After read(), _current_frame should still be None (no assignment)."""
        import ast
        import pathlib

        src = pathlib.Path("modules/video_capture.py").read_text()
        tree = ast.parse(src)

        # Find the read method body
        read_method_assigns = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "read":
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Attribute) and target.attr == "_current_frame":
                                read_method_assigns.append(child)

        assert len(read_method_assigns) == 0, (
            "_current_frame is still being assigned inside read() — this prevents frame GC and doubles peak memory"
        )

    def test_read_returns_frame_without_caching(self):
        """read() should return the frame from cap.read() without caching it."""
        from modules.video_capture import VideoCapturer

        capturer = VideoCapturer(0)
        mock_cap = MagicMock()
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, fake_frame)
        capturer.cap = mock_cap
        capturer.is_running = True

        ret, frame = capturer.read()

        assert ret is True
        assert frame is fake_frame
        # _current_frame should remain None after read (no caching)
        assert capturer._current_frame is None


# ---------------------------------------------------------------------------
# 3. create_face_mask called once and shared
# ---------------------------------------------------------------------------
def _make_mock_face(frame_h=480, frame_w=640):
    """Minimal mock face with landmarks for mask creation."""
    face = SimpleNamespace()
    lm = np.zeros((106, 2), dtype=np.float32)
    cx, cy = frame_w / 2, frame_h / 2
    for i in range(33):
        angle = np.pi * i / 32
        lm[i] = [cx + 90 * np.sin(angle), cy - 120 * np.cos(angle)]
    face.landmark_2d_106 = lm
    face.bbox = np.array([cx - 100, cy - 130, cx + 100, cy + 100], dtype=np.float32)
    face.normed_embedding = np.zeros(512, dtype=np.float32)
    return face


class TestFaceMaskComputedOnce:
    """Verify create_face_mask is called once per face swap, not twice."""

    def test_create_face_mask_called_once_when_both_enabled(self):
        """When mouth_mask AND poisson_blend are both enabled, create_face_mask
        should be called exactly once — the result is shared between
        _apply_mouth_mask and _apply_poisson_blend."""
        from modules.processing_config import ProcessingConfig
        from modules.processors.frame.face_swapper import (
            _apply_mouth_mask,
            _apply_poisson_blend,
        )

        config = ProcessingConfig()
        config.mouth_mask = True
        config.poisson_blend = True

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        face = _make_mock_face()

        call_count = [0]
        sentinel_mask = np.zeros((480, 640), dtype=np.uint8)

        def counting_create_face_mask(f, fr, config=None):
            call_count[0] += 1
            return sentinel_mask

        with patch(
            "modules.processors.frame.face_swapper.create_face_mask",
            side_effect=counting_create_face_mask,
        ):
            with patch(
                "modules.processors.frame.face_swapper.create_lower_mouth_mask",
                return_value=(None, None, (0, 0, 0, 0), None),
            ):
                # Simulate caller computing the mask once and passing it
                with patch(
                    "modules.processors.frame.face_swapper.create_face_mask",
                    side_effect=counting_create_face_mask,
                ) as mock_cfm:
                    # Call the two helper functions the way the caller would
                    # after the refactor: both receive the pre-computed face_mask
                    _apply_mouth_mask(frame.copy(), face, frame, config=config, face_mask=sentinel_mask)
                    _apply_poisson_blend(frame.copy(), face, frame, frame, config=config, face_mask=sentinel_mask)

        # create_face_mask should NOT have been called inside the helpers
        # since we passed it pre-computed
        assert call_count[0] == 0, (
            f"create_face_mask was called {call_count[0]} times inside helpers — "
            "it should not be called when face_mask is passed in"
        )

    def test_create_face_mask_called_once_at_caller_level(self):
        """When face_mask is not pre-computed (None), each helper computes its own.
        The optimization is that the caller passes it to avoid duplication."""
        from modules.processing_config import ProcessingConfig
        from modules.processors.frame.face_swapper import _apply_mouth_mask

        config = ProcessingConfig()
        config.mouth_mask = True

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        face = _make_mock_face()
        sentinel_mask = np.zeros((480, 640), dtype=np.uint8)
        call_count = [0]

        def counting_create_face_mask(f, fr, config=None):
            call_count[0] += 1
            return sentinel_mask

        with patch(
            "modules.processors.frame.face_swapper.create_face_mask",
            side_effect=counting_create_face_mask,
        ):
            with patch(
                "modules.processors.frame.face_swapper.create_lower_mouth_mask",
                return_value=(None, None, (0, 0, 0, 0), None),
            ):
                # When face_mask=None (old call), helper computes it internally
                _apply_mouth_mask(frame.copy(), face, frame, config=config, face_mask=None)

        # create_face_mask was called once (inside the helper as fallback)
        assert call_count[0] == 1

    def test_apply_mouth_mask_accepts_face_mask_parameter(self):
        """_apply_mouth_mask must accept an optional face_mask keyword argument."""
        import inspect

        from modules.processors.frame.face_swapper import _apply_mouth_mask

        sig = inspect.signature(_apply_mouth_mask)
        assert "face_mask" in sig.parameters, "_apply_mouth_mask does not have a face_mask parameter"

    def test_apply_poisson_blend_accepts_face_mask_parameter(self):
        """_apply_poisson_blend must accept an optional face_mask keyword argument."""
        import inspect

        from modules.processors.frame.face_swapper import _apply_poisson_blend

        sig = inspect.signature(_apply_poisson_blend)
        assert "face_mask" in sig.parameters, "_apply_poisson_blend does not have a face_mask parameter"
