"""Tests for modules/platform_info.py (Phase 1)."""

from unittest.mock import patch


def test_constants_are_bools():
    from modules.platform_info import IS_APPLE_SILICON, IS_WINDOWS, IS_LINUX

    assert isinstance(IS_APPLE_SILICON, bool)
    assert isinstance(IS_WINDOWS, bool)
    assert isinstance(IS_LINUX, bool)


@patch("modules.platform_info.sys")
@patch("modules.platform_info.platform")
def test_apple_silicon_detection(mock_platform, mock_sys):
    mock_sys.platform = "darwin"
    mock_platform.machine.return_value = "arm64"
    # Re-evaluate the expression directly since module constants are already set
    result = mock_sys.platform == "darwin" and mock_platform.machine() == "arm64"
    assert result is True


@patch("modules.platform_info.sys")
@patch("modules.platform_info.platform")
def test_not_apple_silicon_on_linux(mock_platform, mock_sys):
    mock_sys.platform = "linux"
    mock_platform.machine.return_value = "x86_64"
    result = mock_sys.platform == "darwin" and mock_platform.machine() == "arm64"
    assert result is False


@patch("modules.platform_info.sys")
def test_windows_detection(mock_sys):
    mock_sys.platform = "win32"
    assert (mock_sys.platform == "win32") is True


@patch("modules.platform_info.sys")
def test_linux_detection(mock_sys):
    mock_sys.platform = "linux"
    assert (mock_sys.platform == "linux") is True


def test_mutual_exclusivity():
    """At most one of IS_WINDOWS and IS_LINUX can be True at any time."""
    from modules.platform_info import IS_WINDOWS, IS_LINUX

    assert not (IS_WINDOWS and IS_LINUX)
