"""Benchmark-specific fixtures — NOT imported by the main conftest.py.

These fixtures skip ML-module stubbing so real ONNX models load properly.
Only activated when running with: pytest -m benchmark or pytest -m integration
"""
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
BASELINES_DIR = PROJECT_ROOT / "benchmarks" / "baselines"


def _have_model(name: str) -> bool:
    return (MODELS_DIR / name).is_file()


# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------
requires_swap_model = pytest.mark.skipif(
    not _have_model("inswapper_128_fp16.onnx"),
    reason="inswapper_128_fp16.onnx not found in models/",
)
requires_enhancer_model = pytest.mark.skipif(
    not _have_model("gfpgan-1024-fp16.onnx") and not _have_model("gfpgan-1024.onnx"),
    reason="gfpgan-1024 model not found in models/",
)


# ---------------------------------------------------------------------------
# Provider auto-detection
# ---------------------------------------------------------------------------
def detect_providers() -> list[str]:
    """Return the best available ONNX Runtime execution providers."""
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
    except ImportError:
        return ["CPUExecutionProvider"]

    if sys.platform == "darwin":
        if "CoreMLExecutionProvider" in available:
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    else:
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


@pytest.fixture(scope="session")
def execution_providers():
    """Auto-detected providers for the current machine."""
    return detect_providers()


# ---------------------------------------------------------------------------
# Synthetic face data
# ---------------------------------------------------------------------------
def _make_landmarks_106() -> np.ndarray:
    """Generate plausible 106-point face landmarks centered in a 480x640 frame."""
    lm = np.zeros((106, 2), dtype=np.float32)

    # Face outline (points 0-32): oval centered at (320, 280)
    cx, cy = 320.0, 280.0
    rx, ry = 90.0, 120.0
    for i in range(33):
        angle = np.pi * i / 32
        lm[i] = [cx + rx * np.sin(angle), cy - ry * np.cos(angle)]

    # Left eyebrow (33-42)
    for i, idx in enumerate(range(33, 43)):
        lm[idx] = [270 + i * 8, 200 - abs(i - 5) * 2]
    # Right eyebrow (43-51)
    for i, idx in enumerate(range(43, 52)):
        lm[idx] = [330 + i * 8, 200 - abs(i - 4) * 2]

    # Lower lip (52-63): oval below nose
    for i, idx in enumerate(range(52, 64)):
        angle = 2 * np.pi * i / 12
        lm[idx] = [320 + 25 * np.cos(angle), 340 + 10 * np.sin(angle)]

    # Nose (64-72)
    for i, idx in enumerate(range(64, 73)):
        lm[idx] = [310 + i * 2.5, 260 + abs(i - 4) * 3]

    # Left eye (73-86)
    for i, idx in enumerate(range(73, 87)):
        angle = 2 * np.pi * i / 14
        lm[idx] = [290 + 15 * np.cos(angle), 240 + 8 * np.sin(angle)]

    # Right eye (87-96)
    for i, idx in enumerate(range(87, 97)):
        angle = 2 * np.pi * i / 10
        lm[idx] = [350 + 15 * np.cos(angle), 240 + 8 * np.sin(angle)]

    # Left eyebrow detailed (97-104)
    for i, idx in enumerate(range(97, 105)):
        lm[idx] = [268 + i * 8, 195 - abs(i - 4) * 2]
    # Remaining point
    lm[105] = [320, 310]

    return lm


def _make_face(embedding_dim: int = 512) -> SimpleNamespace:
    """Create a Face-like namespace with realistic attributes."""
    face = SimpleNamespace()
    emb = np.random.randn(embedding_dim).astype(np.float32)
    face.normed_embedding = emb / np.linalg.norm(emb)
    face.embedding = face.normed_embedding.copy()
    face.kps = np.array(
        [[290, 240], [350, 240], [320, 270], [295, 310], [345, 310]],
        dtype=np.float32,
    )
    face.bbox = np.array([230, 160, 410, 400], dtype=np.float32)
    face.det_score = 0.99
    face.landmark_2d_106 = _make_landmarks_106()
    face.landmark_3d_68 = np.zeros((68, 3), dtype=np.float32)
    face.gender = 1
    face.age = 30
    return face


@pytest.fixture
def synthetic_frame():
    """A 480x640 BGR frame with skin-tone-like gradients."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Skin-tone base
    frame[:, :] = [160, 180, 210]  # BGR skin-ish
    # Add some gradient variation
    for y in range(480):
        for x in range(0, 640, 64):
            noise = np.random.randint(-10, 10, 3)
            x_end = min(x + 64, 640)
            frame[y, x:x_end] = np.clip(frame[y, x:x_end].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame


@pytest.fixture
def synthetic_face():
    """A single synthetic Face object."""
    return _make_face()


@pytest.fixture
def synthetic_faces():
    """Three synthetic Face objects for multi-face testing."""
    faces = []
    for i in range(3):
        f = _make_face()
        # Offset each face horizontally
        offset = (i - 1) * 150
        f.bbox += [offset, 0, offset, 0]
        f.kps[:, 0] += offset
        f.landmark_2d_106[:, 0] += offset
        faces.append(f)
    return faces


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------
class Timer:
    """Collect timing stats over multiple iterations."""

    def __init__(self):
        self.times: list[float] = []

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.times.append(time.perf_counter() - self._start)

    @property
    def stats(self) -> dict:
        if not self.times:
            return {}
        arr = np.array(self.times)
        return {
            "n": len(arr),
            "mean_ms": float(np.mean(arr) * 1000),
            "median_ms": float(np.median(arr) * 1000),
            "std_ms": float(np.std(arr) * 1000),
            "p95_ms": float(np.percentile(arr, 95) * 1000),
            "p99_ms": float(np.percentile(arr, 99) * 1000),
            "min_ms": float(np.min(arr) * 1000),
            "max_ms": float(np.max(arr) * 1000),
            "fps": float(1.0 / np.mean(arr)) if np.mean(arr) > 0 else 0,
        }


@pytest.fixture
def timer():
    return Timer()


# ---------------------------------------------------------------------------
# Baseline save / compare
# ---------------------------------------------------------------------------
class BaselineManager:
    """Manages JSON baseline files for benchmark comparison."""

    def __init__(self, baselines_dir: Path):
        self.baselines_dir = baselines_dir
        self.baselines_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.baselines_dir / f"{name}.json"

    def save(self, name: str, results: dict) -> Path:
        """Save benchmark results as baseline."""
        path = self._path(name)
        data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "providers": results.get("providers", []),
            "results": results,
        }
        path.write_text(json.dumps(data, indent=2))
        return path

    def load(self, name: str) -> dict | None:
        """Load a previously saved baseline."""
        path = self._path(name)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def compare(self, name: str, current: dict, tolerance_pct: float = 5.0) -> dict:
        """Compare current results against baseline.

        Returns a dict with 'passed', 'regressions', and 'improvements'.
        """
        baseline = self.load(name)
        if baseline is None:
            return {"passed": True, "message": "No baseline found, skipping comparison"}

        baseline_results = baseline["results"]
        regressions = []
        improvements = []

        for key in current:
            if key in ("providers", "name"):
                continue
            if key not in baseline_results:
                continue

            cur = current[key]
            base = baseline_results[key]

            if not isinstance(cur, dict) or not isinstance(base, dict):
                continue

            # Compare FPS (higher is better)
            if "fps" in cur and "fps" in base:
                cur_fps = cur["fps"]
                base_fps = base["fps"]
                if base_fps > 0:
                    pct_change = ((cur_fps - base_fps) / base_fps) * 100
                    entry = {
                        "metric": f"{key}.fps",
                        "baseline": base_fps,
                        "current": cur_fps,
                        "change_pct": round(pct_change, 2),
                    }
                    if pct_change < -tolerance_pct:
                        regressions.append(entry)
                    elif pct_change > tolerance_pct:
                        improvements.append(entry)

            # Compare mean_ms (lower is better)
            if "mean_ms" in cur and "mean_ms" in base:
                cur_ms = cur["mean_ms"]
                base_ms = base["mean_ms"]
                if base_ms > 0:
                    pct_change = ((cur_ms - base_ms) / base_ms) * 100
                    entry = {
                        "metric": f"{key}.mean_ms",
                        "baseline": base_ms,
                        "current": cur_ms,
                        "change_pct": round(pct_change, 2),
                    }
                    if pct_change > tolerance_pct:
                        regressions.append(entry)
                    elif pct_change < -tolerance_pct:
                        improvements.append(entry)

        return {
            "passed": len(regressions) == 0,
            "regressions": regressions,
            "improvements": improvements,
        }


@pytest.fixture(scope="session")
def baseline_manager():
    return BaselineManager(BASELINES_DIR)
