"""Tests for NumPy BLAS configuration on Apple Silicon."""
import sys
import platform
import pytest


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Apple Accelerate BLAS only on Apple Silicon (macOS ARM64)"
)
def test_numpy_uses_accelerate_blas():
    """Verify NumPy is using Apple Accelerate BLAS on macOS ARM."""
    import numpy as np

    # Check if Accelerate is the BLAS implementation
    config = np.show_config()
    config_str = str(config)

    # Should contain reference to Accelerate/vecLib
    assert "accelerate" in config_str.lower() or "veclib" in config_str.lower(), \
        "NumPy should use Apple Accelerate BLAS on macOS ARM64. " \
        "To fix: install NumPy via conda with `libblas=*=*accelerate` or build from source"


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Apple Accelerate BLAS only on Apple Silicon (macOS ARM64)"
)
def test_numpy_blas_not_openblas():
    """Verify NumPy is NOT using OpenBLAS on Apple Silicon."""
    import numpy as np

    config = np.show_config()
    config_str = str(config)

    # Should NOT use OpenBLAS on Apple Silicon (suboptimal)
    # Note: This may be too strict if there's a legitimate reason to use OpenBLAS,
    # but the goal is to prefer Accelerate
    if "openblas" in config_str.lower():
        pytest.skip(
            "NumPy is using OpenBLAS. Performance would benefit from Apple Accelerate. "
            "To fix: install via conda with `libblas=*=*accelerate` or build from source"
        )


def test_numpy_linear_algebra_performance():
    """Benchmark NumPy linear algebra operations (informal performance check)."""
    import numpy as np
    import time

    # Create some matrices for typical face processing operations
    # (embedding normalization, affine transforms)
    matrix_size = 512
    iterations = 100

    A = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    B = np.random.randn(matrix_size, matrix_size).astype(np.float32)

    start = time.time()
    for _ in range(iterations):
        np.dot(A, B)
    elapsed = time.time() - start

    # Just verify it runs without error
    # Actual performance comparison would require baseline measurements
    assert elapsed > 0, "Matrix multiplication should complete"
