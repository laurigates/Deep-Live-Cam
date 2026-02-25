"""Tests for download_model_if_needed in modules/utilities.py (Phase 4)."""

from unittest.mock import patch, MagicMock
import os


@patch("modules.utilities.conditional_download")
@patch("modules.utilities.os.path.exists")
def test_returns_true_when_model_exists(mock_exists, mock_download):
    from modules.utilities import download_model_if_needed

    # Model already exists
    mock_exists.return_value = True
    result = download_model_if_needed("model.onnx", ["http://example.com/model.onnx"], "TEST", models_dir="/tmp/models")
    assert result is True
    mock_download.assert_called_once_with("/tmp/models", ["http://example.com/model.onnx"])


@patch("modules.utilities.conditional_download")
@patch("modules.utilities.os.path.exists")
def test_returns_false_when_model_missing_after_download(mock_exists, mock_download):
    from modules.utilities import download_model_if_needed

    # Model doesn't exist before or after download
    mock_exists.return_value = False
    with patch("modules.core.update_status") as mock_status:
        result = download_model_if_needed("model.onnx", ["http://example.com/model.onnx"], "TEST", models_dir="/tmp/models")
    assert result is False


@patch("modules.utilities.conditional_download")
@patch("modules.utilities.os.path.exists")
def test_calls_update_status_when_downloading(mock_exists, mock_download):
    from modules.utilities import download_model_if_needed

    # First call: not exists (triggers download status). Second call: exists (after download).
    mock_exists.side_effect = [False, True]
    with patch("modules.core.update_status") as mock_status:
        result = download_model_if_needed("model.onnx", ["http://example.com/model.onnx"], "TEST", models_dir="/tmp/models")
    assert result is True
    mock_status.assert_called_once_with("Downloading model.onnx...", "TEST")


@patch("modules.utilities.conditional_download")
@patch("modules.utilities.os.path.exists")
def test_uses_default_models_dir(mock_exists, mock_download):
    from modules.utilities import download_model_if_needed

    mock_exists.return_value = True
    result = download_model_if_needed("model.onnx", ["http://example.com/model.onnx"], "TEST")
    assert result is True
    # Should have used the default MODELS_DIR
    call_args = mock_download.call_args[0]
    assert call_args[0].endswith("models")
