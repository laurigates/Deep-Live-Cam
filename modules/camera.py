"""Platform-aware camera enumeration."""

import cv2

from modules.platform_info import IS_APPLE_SILICON, IS_WINDOWS


def get_available_cameras():
    """Returns (indices, names) of available cameras.

    macOS: returns fixed [(0, 1), ("Camera 0", "Camera 1")] — no probing
    to avoid the OBSENSOR segfault (see cross-platform.md).
    Windows: uses pygrabber FilterGraph for named device enumeration.
    Linux: bounded cv2.VideoCapture probe, breaks after 3 consecutive failures.
    """
    if IS_APPLE_SILICON or _is_macos():
        return [0, 1], ["Camera 0", "Camera 1"]

    if IS_WINDOWS:
        return _enumerate_windows()

    return _enumerate_linux()


def _is_macos() -> bool:
    import sys
    return sys.platform == "darwin"


def _enumerate_windows():
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        devices = graph.get_input_devices()
        camera_indices = list(range(len(devices)))
        camera_names = devices

        if not camera_names:
            camera_indices = []
            camera_names = []
            for idx in range(2):
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    camera_indices.append(idx)
                    camera_names.append(f"Camera {idx}")
                    cap.release()

        if not camera_names:
            return [], ["No cameras found"]
        return camera_indices, camera_names
    except Exception as e:
        print(f"Error detecting cameras: {str(e)}")
        return [], ["No cameras found"]


def _enumerate_linux():
    camera_indices = []
    camera_names = []
    consecutive_failures = 0
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            camera_indices.append(i)
            camera_names.append(f"Camera {i}")
            cap.release()
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                break

    if not camera_names:
        return [], ["No cameras found"]
    return camera_indices, camera_names
