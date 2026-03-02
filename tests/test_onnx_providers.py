"""Tests for modules/onnx_providers.py (Phase 2)."""

from unittest.mock import patch
import os


def test_non_coreml_providers_pass_through():
    from modules.onnx_providers import build_providers_config

    result = build_providers_config(["CPUExecutionProvider", "CUDAExecutionProvider"])
    assert result == ["CPUExecutionProvider", "CUDAExecutionProvider"]


@patch("modules.onnx_providers.IS_APPLE_SILICON", True)
@patch("modules.onnx_providers.os.makedirs")
def test_coreml_on_apple_silicon_returns_tuple(mock_makedirs):
    from modules.onnx_providers import build_providers_config

    result = build_providers_config(["CoreMLExecutionProvider"])
    assert len(result) == 1
    assert isinstance(result[0], tuple)
    assert result[0][0] == "CoreMLExecutionProvider"

    opts = result[0][1]
    assert opts["ModelFormat"] == "MLProgram"
    assert opts["MLComputeUnits"] == "ALL"
    assert opts["SpecializationStrategy"] == "FastPrediction"
    assert opts["AllowLowPrecisionAccumulationOnGPU"] == 1
    assert opts["EnableOnSubgraphs"] == 1
    assert "MaximumCacheSize" not in opts  # rejected by onnxruntime-silicon 1.24.2
    assert "ModelCacheDirectory" in opts
    mock_makedirs.assert_called_once()


@patch("modules.onnx_providers.IS_APPLE_SILICON", False)
def test_coreml_not_on_apple_silicon_passes_through():
    from modules.onnx_providers import build_providers_config

    result = build_providers_config(["CoreMLExecutionProvider"])
    assert result == ["CoreMLExecutionProvider"]


@patch("modules.onnx_providers.IS_APPLE_SILICON", True)
@patch("modules.onnx_providers.os.makedirs")
def test_mixed_providers(mock_makedirs):
    from modules.onnx_providers import build_providers_config

    result = build_providers_config(
        ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    )
    assert len(result) == 2
    assert isinstance(result[0], tuple)
    assert result[1] == "CPUExecutionProvider"


def test_empty_providers():
    from modules.onnx_providers import build_providers_config

    result = build_providers_config([])
    assert result == []


@patch("modules.onnx_providers.IS_APPLE_SILICON", True)
@patch("modules.onnx_providers.os.makedirs")
def test_options_dict_has_all_expected_keys(mock_makedirs):
    """Regression test: all expected CoreML options are present."""
    from modules.onnx_providers import build_providers_config

    result = build_providers_config(["CoreMLExecutionProvider"])
    opts = result[0][1]
    expected_keys = {
        "ModelFormat",
        "MLComputeUnits",
        "SpecializationStrategy",
        "AllowLowPrecisionAccumulationOnGPU",
        "EnableOnSubgraphs",
        "ModelCacheDirectory",
    }
    assert set(opts.keys()) == expected_keys


@patch("modules.onnx_providers.IS_APPLE_SILICON", True)
@patch("modules.onnx_providers.os.makedirs")
def test_coreml_compute_units_override(mock_makedirs):
    """Explicitly passed coreml_compute_units overrides the default."""
    from modules.onnx_providers import build_providers_config

    result = build_providers_config(
        ["CoreMLExecutionProvider"], coreml_compute_units="CPUAndGPU"
    )
    assert result[0][1]["MLComputeUnits"] == "CPUAndGPU"


@patch("modules.onnx_providers.IS_APPLE_SILICON", True)
@patch("modules.onnx_providers.os.makedirs")
def test_coreml_compute_units_cpu_only(mock_makedirs):
    """CPUOnly compute units disables GPU and ANE routing."""
    from modules.onnx_providers import build_providers_config

    result = build_providers_config(
        ["CoreMLExecutionProvider"], coreml_compute_units="CPUOnly"
    )
    assert result[0][1]["MLComputeUnits"] == "CPUOnly"


@patch("modules.onnx_providers.IS_APPLE_SILICON", True)
@patch("modules.onnx_providers.os.makedirs")
def test_invalid_compute_units_falls_back_to_all(mock_makedirs):
    """Invalid compute unit values fall back to 'ALL'."""
    from modules.onnx_providers import build_providers_config

    result = build_providers_config(
        ["CoreMLExecutionProvider"], coreml_compute_units="InvalidValue"
    )
    assert result[0][1]["MLComputeUnits"] == "ALL"


@patch("modules.onnx_providers.IS_APPLE_SILICON", True)
@patch("modules.onnx_providers.os.makedirs")
def test_coreml_compute_units_reads_from_globals(mock_makedirs):
    """Without explicit override, reads coreml_compute_units from modules.globals."""
    from modules.onnx_providers import build_providers_config
    import modules.globals

    original = modules.globals.coreml_compute_units
    try:
        modules.globals.coreml_compute_units = "CPUAndGPU"
        result = build_providers_config(["CoreMLExecutionProvider"])
        assert result[0][1]["MLComputeUnits"] == "CPUAndGPU"
    finally:
        modules.globals.coreml_compute_units = original
