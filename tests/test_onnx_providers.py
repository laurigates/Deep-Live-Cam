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
