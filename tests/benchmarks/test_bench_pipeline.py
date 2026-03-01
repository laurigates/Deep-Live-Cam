"""End-to-end pipeline benchmarks: swap + enhancement + masking.

Run with: uv run pytest tests/benchmarks/test_bench_pipeline.py -m benchmark -v
"""

import time

import numpy as np
import psutil
import pytest

from tests.benchmarks.conftest import (
    Timer,
    _make_face,
    requires_enhancer_model,
    requires_swap_model,
)

pytestmark = [pytest.mark.benchmark, pytest.mark.integration]

WARMUP_ITERS = 3
BENCH_ITERS = 30


def _setup_globals(providers: list[str], **overrides) -> None:
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
    modules.globals.fp_ui = {
        "face_enhancer": False,
        "face_enhancer_gpen256": False,
        "face_enhancer_gpen512": False,
        "face_enhancer_codeformer": False,
    }

    for key, value in overrides.items():
        setattr(modules.globals, key, value)


def _measure_memory() -> float:
    """Return current process RSS in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


# ---------------------------------------------------------------------------
# Full pipeline: swap → enhance
# ---------------------------------------------------------------------------
@requires_swap_model
@requires_enhancer_model
class TestFullPipeline:
    """Benchmark the complete processing pipeline."""

    def test_swap_then_enhance(self, execution_providers, timer, synthetic_frame, synthetic_face, baseline_manager):
        """Swap + enhancement: the most common live-mode pipeline."""
        _setup_globals(execution_providers)
        import modules.globals

        modules.globals.fp_ui["face_enhancer"] = True

        from modules.processors.frame.face_enhancer import enhance_face, get_face_enhancer
        from modules.processors.frame.face_swapper import swap_face

        get_face_enhancer(providers=execution_providers)

        source_face = _make_face()
        frame = synthetic_frame.copy()

        for _ in range(WARMUP_ITERS):
            swapped = swap_face(source_face, synthetic_face, frame.copy())
            enhance_face(synthetic_face, swapped)

        mem_before = _measure_memory()

        for _ in range(BENCH_ITERS):
            with timer:
                swapped = swap_face(source_face, synthetic_face, frame.copy())
                enhanced = enhance_face(synthetic_face, swapped)

        mem_after = _measure_memory()
        stats = timer.stats

        results = {
            "name": "swap_then_enhance",
            "providers": execution_providers,
            "swap_then_enhance": stats,
            "memory_delta_mb": round(mem_after - mem_before, 2),
        }

        print("\n--- Swap + Enhance Pipeline ---")
        print(f"  Providers: {execution_providers}")
        print(f"  Mean: {stats['mean_ms']:.2f} ms")
        print(f"  Median: {stats['median_ms']:.2f} ms")
        print(f"  P95: {stats['p95_ms']:.2f} ms")
        print(f"  FPS: {stats['fps']:.1f}")
        print(f"  Memory delta: {mem_after - mem_before:.1f} MB")

        comparison = baseline_manager.compare("swap_then_enhance", results)
        if comparison.get("regressions"):
            for r in comparison["regressions"]:
                print(
                    f"  REGRESSION: {r['metric']}: {r['baseline']:.2f} -> {r['current']:.2f} ({r['change_pct']:+.1f}%)"
                )

        assert stats["mean_ms"] > 0

    def test_swap_enhance_mouth_mask(self, execution_providers, timer, synthetic_frame, synthetic_face):
        """Full pipeline with mouth masking enabled."""
        _setup_globals(execution_providers, mouth_mask=True)
        import modules.globals

        modules.globals.fp_ui["face_enhancer"] = True

        from modules.processors.frame.face_enhancer import enhance_face, get_face_enhancer
        from modules.processors.frame.face_swapper import swap_face

        get_face_enhancer(providers=execution_providers)

        source_face = _make_face()
        frame = synthetic_frame.copy()

        for _ in range(WARMUP_ITERS):
            swapped = swap_face(source_face, synthetic_face, frame.copy())
            enhance_face(synthetic_face, swapped)

        for _ in range(BENCH_ITERS):
            with timer:
                swapped = swap_face(source_face, synthetic_face, frame.copy())
                enhanced = enhance_face(synthetic_face, swapped)

        stats = timer.stats
        print("\n--- Swap + Enhance + Mouth Mask ---")
        print(f"  Mean: {stats['mean_ms']:.2f} ms  |  FPS: {stats['fps']:.1f}")


# ---------------------------------------------------------------------------
# Toggle matrix: measure incremental cost of each toggle
# ---------------------------------------------------------------------------
@requires_swap_model
class TestToggleIncrementalCost:
    """Measure the incremental cost of each processing toggle."""

    TOGGLE_CONFIGS = [
        ("baseline", {}),
        ("mouth_mask", {"mouth_mask": True}),
        ("poisson_blend", {"poisson_blend": True}),
        ("color_correction", {"color_correction": True}),
        ("sharpness", {"sharpness": 0.5}),
        ("prepaste_off", {"prepaste_upscale": False}),
        ("opacity_blend", {"opacity": 0.7}),
        ("interpolation", {"enable_interpolation": True, "interpolation_weight": 0.2}),
        (
            "all_on",
            {
                "mouth_mask": True,
                "poisson_blend": True,
                "color_correction": True,
                "sharpness": 0.5,
                "opacity": 0.8,
            },
        ),
    ]

    @pytest.mark.parametrize(
        "config_name,overrides",
        TOGGLE_CONFIGS,
        ids=[c[0] for c in TOGGLE_CONFIGS],
    )
    def test_toggle_cost(self, execution_providers, timer, synthetic_frame, synthetic_face, config_name, overrides):
        """Measure swap latency with specific toggle configuration."""
        _setup_globals(execution_providers, **overrides)

        from modules.processors.frame.face_swapper import swap_face

        source_face = _make_face()
        frame = synthetic_frame.copy()

        for _ in range(WARMUP_ITERS):
            swap_face(source_face, synthetic_face, frame.copy())

        for _ in range(BENCH_ITERS):
            with timer:
                swap_face(source_face, synthetic_face, frame.copy())

        stats = timer.stats
        print(f"\n--- Toggle: {config_name} ---")
        print(f"  Mean: {stats['mean_ms']:.2f} ms  |  FPS: {stats['fps']:.1f}")


# ---------------------------------------------------------------------------
# Memory stability over sustained runs
# ---------------------------------------------------------------------------
@requires_swap_model
class TestMemoryStability:
    """Verify no memory leaks over extended runs."""

    def test_no_memory_leak_300_frames(self, execution_providers, synthetic_frame, synthetic_face):
        """Run 300 frames and check memory doesn't grow significantly."""
        _setup_globals(execution_providers)

        from modules.processors.frame.face_swapper import swap_face

        source_face = _make_face()
        frame = synthetic_frame.copy()

        # Warmup
        for _ in range(10):
            swap_face(source_face, synthetic_face, frame.copy())

        mem_start = _measure_memory()

        for i in range(300):
            swap_face(source_face, synthetic_face, frame.copy())

        mem_end = _measure_memory()
        growth_mb = mem_end - mem_start

        print("\n--- Memory Stability (300 frames) ---")
        print(f"  Start: {mem_start:.1f} MB")
        print(f"  End: {mem_end:.1f} MB")
        print(f"  Growth: {growth_mb:.1f} MB")

        # Allow up to 100 MB growth (models, caches, etc.)
        # This is generous; the key is that it doesn't grow unbounded.
        assert growth_mb < 100, f"Memory grew by {growth_mb:.1f} MB over 300 frames"
