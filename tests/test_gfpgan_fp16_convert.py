"""Tests for FP16 GFPGAN model preference and conversion script."""
import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


def test_get_face_enhancer_prefers_fp16(tmp_path):
    """get_face_enhancer should prefer the FP16 model when both exist."""
    from modules.processors.frame import face_enhancer

    fp16_path = str(tmp_path / "gfpgan-1024-fp16.onnx")
    fp32_path = str(tmp_path / "gfpgan-1024.onnx")

    # Create dummy files
    open(fp16_path, "w").close()
    open(fp32_path, "w").close()

    # Reset singleton
    face_enhancer.FACE_ENHANCER = None

    with patch.object(face_enhancer, "MODELS_DIR", str(tmp_path)):
        with patch("onnxruntime.InferenceSession") as mock_session:
            mock_inst = MagicMock()
            mock_input = MagicMock()
            mock_input.name = "input"
            mock_input.shape = [1, 3, 512, 512]
            mock_input.type = "tensor(float)"
            mock_inst.get_inputs.return_value = [mock_input]
            mock_output = MagicMock()
            mock_output.name = "output"
            mock_output.shape = [1, 3, 1024, 1024]
            mock_output.type = "tensor(float)"
            mock_inst.get_outputs.return_value = [mock_output]
            mock_inst.get_providers.return_value = ["CPUExecutionProvider"]
            mock_session.return_value = mock_inst

            face_enhancer.get_face_enhancer()

            # Verify FP16 path was used
            call_args = mock_session.call_args
            assert "fp16" in call_args[0][0]

    # Clean up singleton
    face_enhancer.FACE_ENHANCER = None


def test_get_face_enhancer_falls_back_to_fp32(tmp_path):
    """get_face_enhancer should fall back to FP32 when FP16 doesn't exist."""
    from modules.processors.frame import face_enhancer

    fp32_path = str(tmp_path / "gfpgan-1024.onnx")
    open(fp32_path, "w").close()

    face_enhancer.FACE_ENHANCER = None

    with patch.object(face_enhancer, "MODELS_DIR", str(tmp_path)):
        with patch("onnxruntime.InferenceSession") as mock_session:
            mock_inst = MagicMock()
            mock_input = MagicMock()
            mock_input.name = "input"
            mock_input.shape = [1, 3, 512, 512]
            mock_input.type = "tensor(float)"
            mock_inst.get_inputs.return_value = [mock_input]
            mock_output = MagicMock()
            mock_output.name = "output"
            mock_output.shape = [1, 3, 1024, 1024]
            mock_output.type = "tensor(float)"
            mock_inst.get_outputs.return_value = [mock_output]
            mock_inst.get_providers.return_value = ["CPUExecutionProvider"]
            mock_session.return_value = mock_inst

            face_enhancer.get_face_enhancer()

            call_args = mock_session.call_args
            assert "fp16" not in call_args[0][0]
            assert "gfpgan-1024.onnx" in call_args[0][0]

    face_enhancer.FACE_ENHANCER = None


def test_pre_check_accepts_fp16_model(tmp_path):
    """pre_check should succeed when only FP16 model exists."""
    from modules.processors.frame import face_enhancer

    fp16_path = str(tmp_path / "gfpgan-1024-fp16.onnx")
    open(fp16_path, "w").close()

    with patch.object(face_enhancer, "MODELS_DIR", str(tmp_path)):
        with patch.object(face_enhancer, "conditional_download"):
            result = face_enhancer.pre_check()
            assert result is True


def test_conversion_script_exists():
    """The conversion script should exist."""
    assert os.path.exists("scripts/convert_gfpgan_fp16.py")
