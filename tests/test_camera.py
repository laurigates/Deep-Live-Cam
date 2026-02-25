"""Tests for modules/camera.py (Phase 8)."""

from unittest.mock import patch, MagicMock


@patch("modules.camera.IS_APPLE_SILICON", True)
def test_macos_arm_returns_fixed_cameras():
    from modules.camera import get_available_cameras

    indices, names = get_available_cameras()
    assert indices == [0, 1]
    assert names == ["Camera 0", "Camera 1"]


@patch("modules.camera.IS_APPLE_SILICON", False)
@patch("modules.camera._is_macos", return_value=True)
def test_macos_intel_returns_fixed_cameras(mock_is_macos):
    from modules.camera import get_available_cameras

    indices, names = get_available_cameras()
    assert indices == [0, 1]
    assert names == ["Camera 0", "Camera 1"]


@patch("modules.camera.IS_APPLE_SILICON", False)
@patch("modules.camera._is_macos", return_value=False)
@patch("modules.camera.IS_WINDOWS", False)
@patch("modules.camera.cv2")
def test_linux_bounded_loop(mock_cv2, mock_is_macos):
    """Linux: bounded probe with 3 consecutive failure break."""
    # Camera 0 succeeds, camera 1 succeeds, cameras 2-4 fail (3 consecutive)
    mock_cap = MagicMock()
    open_sequence = [True, True, False, False, False]
    call_count = [0]

    def mock_video_capture(i):
        cap = MagicMock()
        cap.isOpened.return_value = open_sequence[min(i, len(open_sequence) - 1)]
        return cap

    mock_cv2.VideoCapture = mock_video_capture

    from modules.camera import _enumerate_linux

    indices, names = _enumerate_linux()
    assert indices == [0, 1]
    assert names == ["Camera 0", "Camera 1"]


@patch("modules.camera.IS_APPLE_SILICON", False)
@patch("modules.camera._is_macos", return_value=False)
@patch("modules.camera.IS_WINDOWS", False)
@patch("modules.camera.cv2")
def test_linux_no_cameras(mock_cv2, mock_is_macos):
    """Linux: no cameras found returns empty with message."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_cv2.VideoCapture.return_value = mock_cap

    from modules.camera import _enumerate_linux

    indices, names = _enumerate_linux()
    assert indices == []
    assert names == ["No cameras found"]
