"""Performance benchmarks for face enhancement inference.

Run with: uv run pytest tests/benchmarks/test_bench_enhancer.py -m benchmark -v
"""

import numpy as np
import pytest

from tests.benchmarks.conftest import (
    Timer,
    _make_face,
    detect_providers,
    requires_enhancer_model,
)

pytestmark = [pytest.mark.benchmark, pytest.mark.integration]

WARMUP_ITERS = 3
BENCH_ITERS = 30


def _setup_globals(providers: list[str]) -> None:
    import modules.globals

    modules.globals.execution_providers = providers
    modules.globals.fp_ui = {
        "face_enhancer": True,
        "face_enhancer_gpen256": False,
        "face_enhancer_gpen512": False,
        "face_enhancer_codeformer": False,
    }


@requires_enhancer_model
class TestFaceEnhancerInference:
    """Benchmark GFPGAN face enhancement."""

    def test_single_face_enhance_latency(
        self, execution_providers, timer, synthetic_frame, synthetic_face, baseline_manager
    ):
        """Measure per-face enhancement latency."""
        _setup_globals(execution_providers)

        from modules.processors.frame.face_enhancer import enhance_face, get_face_enhancer

        # Force load
        get_face_enhancer(providers=execution_providers)

        frame = synthetic_frame.copy()
        face = synthetic_face

        # Warmup
        for _ in range(WARMUP_ITERS):
            enhance_face(face, frame.copy())

        # Benchmark
        for _ in range(BENCH_ITERS):
            with timer:
                enhance_face(face, frame.copy())

        stats = timer.stats
        results = {
            "name": "single_face_enhance",
            "providers": execution_providers,
            "single_face_enhance": stats,
        }

        print("\n--- Single Face Enhancement ---")
        print(f"  Providers: {execution_providers}")
        print(f"  Mean: {stats['mean_ms']:.2f} ms")
        print(f"  Median: {stats['median_ms']:.2f} ms")
        print(f"  P95: {stats['p95_ms']:.2f} ms")
        print(f"  FPS: {stats['fps']:.1f}")

        comparison = baseline_manager.compare("single_face_enhance", results)
        if comparison.get("regressions"):
            for r in comparison["regressions"]:
                print(
                    f"  REGRESSION: {r['metric']}: {r['baseline']:.2f} -> {r['current']:.2f} ({r['change_pct']:+.1f}%)"
                )

        assert stats["mean_ms"] > 0

    def test_multi_face_enhance_latency(self, execution_providers, timer, synthetic_frame, synthetic_faces):
        """Measure enhancement latency for 3 faces sequentially."""
        _setup_globals(execution_providers)

        from modules.processors.frame.face_enhancer import enhance_face, get_face_enhancer

        get_face_enhancer(providers=execution_providers)

        frame = synthetic_frame.copy()

        for _ in range(WARMUP_ITERS):
            for face in synthetic_faces:
                enhance_face(face, frame.copy())

        for _ in range(BENCH_ITERS):
            result = frame.copy()
            with timer:
                for face in synthetic_faces:
                    result = enhance_face(face, result)

        stats = timer.stats
        print("\n--- Multi-Face Enhancement (3 faces) ---")
        print(f"  Mean: {stats['mean_ms']:.2f} ms  |  FPS: {stats['fps']:.1f}")

    @pytest.mark.parametrize(
        "enhance_size",
        [128, 192, 256, 384, 512],
        ids=["128px", "192px", "256px", "384px", "512px"],
    )
    def test_enhance_size_scaling(self, execution_providers, timer, synthetic_frame, synthetic_face, enhance_size):
        """Measure how live_enhance_size affects enhancement speed."""
        _setup_globals(execution_providers)
        import modules.globals

        modules.globals.live_enhance_size = enhance_size

        from modules.processors.frame.face_enhancer import enhance_face, get_face_enhancer

        get_face_enhancer(providers=execution_providers)

        frame = synthetic_frame.copy()

        for _ in range(WARMUP_ITERS):
            enhance_face(synthetic_face, frame.copy())

        for _ in range(BENCH_ITERS):
            with timer:
                enhance_face(synthetic_face, frame.copy())

        stats = timer.stats
        print(f"\n--- Enhancement @ {enhance_size}px ---")
        print(f"  Mean: {stats['mean_ms']:.2f} ms  |  FPS: {stats['fps']:.1f}")
