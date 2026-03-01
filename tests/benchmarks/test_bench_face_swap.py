"""Performance benchmarks for face swap inference.

Run with: uv run pytest tests/benchmarks/ -m benchmark -v --no-header -rN
Save baseline: uv run pytest tests/benchmarks/ -m benchmark --save-baseline
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.benchmarks.conftest import (
    MODELS_DIR,
    Timer,
    _make_face,
    detect_providers,
    requires_swap_model,
)

pytestmark = [pytest.mark.benchmark, pytest.mark.integration]

# Number of inference iterations per benchmark
WARMUP_ITERS = 3
BENCH_ITERS = 50


def _setup_globals(providers: list[str]) -> None:
    """Configure modules.globals for benchmark runs."""
    import modules.globals

    modules.globals.execution_providers = providers
    modules.globals.many_faces = False
    modules.globals.map_faces = False
    modules.globals.mouth_mask = False
    modules.globals.poisson_blend = False
    modules.globals.color_correction = False
    modules.globals.opacity = 1.0
    modules.globals.sharpness = 0.0
    modules.globals.prepaste_upscale = True
    modules.globals.enable_interpolation = False
    modules.globals.mask_feather_ratio = 12
    modules.globals.face_mask_blur = 31


def _get_face_swapper(providers: list[str]):
    """Load the face swapper singleton with given providers."""
    import modules.processors.frame.face_swapper as swapper_mod

    # Clear singleton so it reloads with new providers
    swapper_mod.FACE_SWAPPER = None
    return swapper_mod.get_face_swapper(providers=providers)


# ---------------------------------------------------------------------------
# Single-face swap
# ---------------------------------------------------------------------------
@requires_swap_model
class TestFaceSwapInference:
    """Benchmark raw face swap inference (no post-processing)."""

    def test_single_face_swap_latency(
        self, execution_providers, timer, synthetic_frame, synthetic_face, baseline_manager
    ):
        """Measure per-frame latency for single-face swap."""
        _setup_globals(execution_providers)

        from modules.processors.frame.face_swapper import swap_face

        source_face = _make_face()
        target_face = synthetic_face
        frame = synthetic_frame.copy()

        # Warmup
        for _ in range(WARMUP_ITERS):
            swap_face(source_face, target_face, frame.copy())

        # Benchmark
        for _ in range(BENCH_ITERS):
            with timer:
                swap_face(source_face, target_face, frame.copy())

        stats = timer.stats
        results = {
            "name": "single_face_swap",
            "providers": execution_providers,
            "single_face_swap": stats,
        }

        # Print results
        print("\n--- Single Face Swap ---")
        print(f"  Providers: {execution_providers}")
        print(f"  Mean: {stats['mean_ms']:.2f} ms")
        print(f"  Median: {stats['median_ms']:.2f} ms")
        print(f"  P95: {stats['p95_ms']:.2f} ms")
        print(f"  P99: {stats['p99_ms']:.2f} ms")
        print(f"  FPS: {stats['fps']:.1f}")

        # Compare against baseline
        comparison = baseline_manager.compare("single_face_swap", results)
        if comparison.get("regressions"):
            for r in comparison["regressions"]:
                print(
                    f"  REGRESSION: {r['metric']}: {r['baseline']:.2f} -> {r['current']:.2f} ({r['change_pct']:+.1f}%)"
                )

        assert stats["mean_ms"] > 0

    def test_multi_face_swap_latency(
        self, execution_providers, timer, synthetic_frame, synthetic_faces, baseline_manager
    ):
        """Measure per-frame latency for 3-face batch swap."""
        _setup_globals(execution_providers)
        import modules.globals

        modules.globals.many_faces = True

        from modules.processors.frame.face_swapper import swap_face

        source_face = _make_face()
        frame = synthetic_frame.copy()

        # Warmup
        for _ in range(WARMUP_ITERS):
            for target_face in synthetic_faces:
                swap_face(source_face, target_face, frame.copy())

        # Benchmark
        for _ in range(BENCH_ITERS):
            result = frame.copy()
            with timer:
                for target_face in synthetic_faces:
                    result = swap_face(source_face, target_face, result)

        stats = timer.stats
        results = {
            "name": "multi_face_swap_3",
            "providers": execution_providers,
            "multi_face_swap_3": stats,
        }

        print("\n--- Multi-Face Swap (3 faces) ---")
        print(f"  Mean: {stats['mean_ms']:.2f} ms")
        print(f"  FPS: {stats['fps']:.1f}")

        comparison = baseline_manager.compare("multi_face_swap_3", results)
        if comparison.get("regressions"):
            for r in comparison["regressions"]:
                print(
                    f"  REGRESSION: {r['metric']}: {r['baseline']:.2f} -> {r['current']:.2f} ({r['change_pct']:+.1f}%)"
                )

        assert stats["mean_ms"] > 0


# ---------------------------------------------------------------------------
# Swap + post-processing pipeline
# ---------------------------------------------------------------------------
@requires_swap_model
class TestSwapWithPostProcessing:
    """Benchmark swap with various post-processing toggles enabled."""

    @pytest.mark.parametrize(
        "toggle_name,toggle_attr,toggle_value",
        [
            ("mouth_mask", "mouth_mask", True),
            ("poisson_blend", "poisson_blend", True),
            ("color_correction", "color_correction", True),
            ("sharpness_0.5", "sharpness", 0.5),
            ("prepaste_upscale", "prepaste_upscale", True),
            ("opacity_0.8", "opacity", 0.8),
        ],
        ids=lambda x: x if isinstance(x, str) else "",
    )
    def test_swap_with_toggle(
        self, execution_providers, timer, synthetic_frame, synthetic_face, toggle_name, toggle_attr, toggle_value
    ):
        """Measure swap latency with a single post-processing toggle on."""
        _setup_globals(execution_providers)
        import modules.globals

        setattr(modules.globals, toggle_attr, toggle_value)

        from modules.processors.frame.face_swapper import swap_face

        source_face = _make_face()
        frame = synthetic_frame.copy()

        # Warmup
        for _ in range(WARMUP_ITERS):
            swap_face(source_face, synthetic_face, frame.copy())

        # Benchmark
        for _ in range(BENCH_ITERS):
            with timer:
                swap_face(source_face, synthetic_face, frame.copy())

        stats = timer.stats
        print(f"\n--- Swap + {toggle_name} ---")
        print(f"  Mean: {stats['mean_ms']:.2f} ms  |  FPS: {stats['fps']:.1f}")

    def test_swap_all_toggles(self, execution_providers, timer, synthetic_frame, synthetic_face):
        """Worst case: swap with all post-processing enabled."""
        _setup_globals(execution_providers)
        import modules.globals

        modules.globals.mouth_mask = True
        modules.globals.poisson_blend = True
        modules.globals.color_correction = True
        modules.globals.sharpness = 0.5
        modules.globals.prepaste_upscale = True
        modules.globals.opacity = 0.8

        from modules.processors.frame.face_swapper import swap_face

        source_face = _make_face()
        frame = synthetic_frame.copy()

        for _ in range(WARMUP_ITERS):
            swap_face(source_face, synthetic_face, frame.copy())

        for _ in range(BENCH_ITERS):
            with timer:
                swap_face(source_face, synthetic_face, frame.copy())

        stats = timer.stats
        print("\n--- Swap (ALL toggles ON) ---")
        print(f"  Mean: {stats['mean_ms']:.2f} ms  |  FPS: {stats['fps']:.1f}")
        print(f"  P95: {stats['p95_ms']:.2f} ms  |  P99: {stats['p99_ms']:.2f} ms")


# ---------------------------------------------------------------------------
# Resolution sweep
# ---------------------------------------------------------------------------
@requires_swap_model
class TestResolutionScaling:
    """Measure how swap performance scales with frame resolution."""

    @pytest.mark.parametrize(
        "width,height",
        [(320, 240), (640, 480), (1280, 720), (1920, 1080)],
        ids=["320x240", "640x480", "720p", "1080p"],
    )
    def test_resolution(self, execution_providers, timer, width, height):
        _setup_globals(execution_providers)

        from modules.processors.frame.face_swapper import swap_face

        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        source_face = _make_face()
        target_face = _make_face()
        # Scale landmarks to match frame dimensions
        scale_x = width / 640.0
        scale_y = height / 480.0
        target_face.bbox = np.array([230 * scale_x, 160 * scale_y, 410 * scale_x, 400 * scale_y], dtype=np.float32)
        target_face.kps[:, 0] *= scale_x
        target_face.kps[:, 1] *= scale_y
        target_face.landmark_2d_106[:, 0] *= scale_x
        target_face.landmark_2d_106[:, 1] *= scale_y

        for _ in range(WARMUP_ITERS):
            swap_face(source_face, target_face, frame.copy())

        for _ in range(BENCH_ITERS // 2):  # Fewer iters for large resolutions
            with timer:
                swap_face(source_face, target_face, frame.copy())

        stats = timer.stats
        print(f"\n--- {width}x{height} ---")
        print(f"  Mean: {stats['mean_ms']:.2f} ms  |  FPS: {stats['fps']:.1f}")
