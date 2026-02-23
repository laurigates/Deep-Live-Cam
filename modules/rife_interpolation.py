"""RIFE frame interpolation using rife-ncnn-vulkan.

Wraps the rife-ncnn-vulkan binary to perform AI-based frame interpolation
on processed video frames. Supports Practical-RIFE v4.25 and v4.25.lite models.

The binary and models are expected to be either:
  1. On the system PATH (user-installed), or
  2. In models/rife-ncnn-vulkan/ (local install)

Models ship with the rife-ncnn-vulkan release archives.
See: https://github.com/nihui/rife-ncnn-vulkan/releases
"""

import glob
import os
import shutil
import subprocess
import sys
from typing import List, Optional

import modules.globals
from modules.paths import MODELS_DIR

NAME = "DLC.RIFE-INTERPOLATION"

RIFE_DIR = os.path.join(MODELS_DIR, "rife-ncnn-vulkan")

AVAILABLE_MODELS = {
    "rife-v4.25": "Practical-RIFE v4.25 (higher quality)",
    "rife-v4.25-lite": "Practical-RIFE v4.25 lite (faster)",
}

DEFAULT_MODEL = "rife-v4.25-lite"


def _update_status(message: str) -> None:
    """Print status and forward to UI if available."""
    print(f"[{NAME}] {message}")
    try:
        from modules.core import update_status

        update_status(message, NAME)
    except Exception:
        pass


def _binary_name() -> str:
    """Return platform-appropriate binary name."""
    if sys.platform == "win32":
        return "rife-ncnn-vulkan.exe"
    return "rife-ncnn-vulkan"


def find_binary() -> Optional[str]:
    """Find rife-ncnn-vulkan binary on PATH or in models directory."""
    # Check PATH first
    path_binary = shutil.which("rife-ncnn-vulkan")
    if path_binary:
        return path_binary

    # Check models directory
    local_binary = os.path.join(RIFE_DIR, _binary_name())
    if os.path.isfile(local_binary) and os.access(local_binary, os.X_OK):
        return local_binary

    return None


def find_model_dir(model_name: str) -> Optional[str]:
    """Find model directory for the given model name.

    Searches in order:
      1. models/rife-ncnn-vulkan/<model_name>/
      2. models/<model_name>/
    """
    # Check in RIFE_DIR (bundled with binary download)
    model_path = os.path.join(RIFE_DIR, model_name)
    if os.path.isdir(model_path):
        return model_path

    # Check in MODELS_DIR directly
    model_path = os.path.join(MODELS_DIR, model_name)
    if os.path.isdir(model_path):
        return model_path

    return None


def pre_check() -> bool:
    """Verify rife-ncnn-vulkan binary and model files exist.

    Returns True if RIFE interpolation is ready to use.
    Returns True (pass-through) when RIFE is disabled.
    """
    if not getattr(modules.globals, "rife_enabled", False):
        return True

    binary = find_binary()
    if not binary:
        _update_status(
            "rife-ncnn-vulkan not found. Install it or place it in "
            "models/rife-ncnn-vulkan/. "
            "See https://github.com/nihui/rife-ncnn-vulkan/releases"
        )
        return False

    model_name = getattr(modules.globals, "rife_model", DEFAULT_MODEL)
    model_dir = find_model_dir(model_name)
    if not model_dir:
        _update_status(
            f"RIFE model '{model_name}' not found. Place model files in "
            f"models/rife-ncnn-vulkan/{model_name}/ or models/{model_name}/"
        )
        return False

    return True


def _count_frames(directory: str) -> int:
    """Count image files in a directory."""
    count = 0
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        count += len(glob.glob(os.path.join(directory, ext)))
    return count


def _build_command(
    binary: str,
    input_dir: str,
    output_dir: str,
    model_dir: str,
    input_frame_count: int,
    multiplier: int,
) -> List[str]:
    """Build the rife-ncnn-vulkan command line."""
    target_frames = input_frame_count * multiplier

    cmd = [
        binary,
        "-i",
        input_dir,
        "-o",
        output_dir,
        "-m",
        model_dir,
        "-n",
        str(target_frames),
        "-f",
        "%04d.jpg",
        "-g",
        "auto",
    ]
    return cmd


def interpolate_frames(temp_directory_path: str) -> Optional[int]:
    """Run RIFE frame interpolation on extracted video frames.

    Interpolates frames in-place: the original frames in temp_directory_path
    are replaced with the interpolated output.

    Args:
        temp_directory_path: Directory containing numbered frame images
            (e.g., 0001.jpg, 0002.jpg, ...)

    Returns:
        The new frame count after interpolation, or None if interpolation
        failed or was skipped.
    """
    binary = find_binary()
    if not binary:
        _update_status("rife-ncnn-vulkan binary not found, skipping interpolation")
        return None

    model_name = getattr(modules.globals, "rife_model", DEFAULT_MODEL)
    model_dir = find_model_dir(model_name)
    if not model_dir:
        _update_status(f"RIFE model '{model_name}' not found, skipping interpolation")
        return None

    multiplier = getattr(modules.globals, "rife_multiplier", 2)
    if multiplier < 2:
        multiplier = 2

    input_frame_count = _count_frames(temp_directory_path)
    if input_frame_count < 2:
        _update_status("Not enough frames for interpolation (need at least 2)")
        return None

    # Create a temporary output directory alongside the input
    rife_output_dir = temp_directory_path + "_rife"
    os.makedirs(rife_output_dir, exist_ok=True)

    cmd = _build_command(
        binary, temp_directory_path, rife_output_dir, model_dir,
        input_frame_count, multiplier,
    )

    _update_status(
        f"Running RIFE interpolation ({model_name}, {multiplier}x) "
        f"on {input_frame_count} frames..."
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout for very long videos
        )
        if result.returncode != 0:
            stderr = result.stderr.strip() if result.stderr else "unknown error"
            _update_status(f"RIFE interpolation failed: {stderr}")
            shutil.rmtree(rife_output_dir, ignore_errors=True)
            return None
    except subprocess.TimeoutExpired:
        _update_status("RIFE interpolation timed out")
        shutil.rmtree(rife_output_dir, ignore_errors=True)
        return None
    except FileNotFoundError:
        _update_status(f"rife-ncnn-vulkan binary not found at {binary}")
        shutil.rmtree(rife_output_dir, ignore_errors=True)
        return None
    except Exception as e:
        _update_status(f"RIFE interpolation error: {e}")
        shutil.rmtree(rife_output_dir, ignore_errors=True)
        return None

    # Count output frames
    output_frame_count = _count_frames(rife_output_dir)
    if output_frame_count == 0:
        _update_status("RIFE produced no output frames")
        shutil.rmtree(rife_output_dir, ignore_errors=True)
        return None

    # Rename output frames to sequential %04d.jpg format (in case they differ)
    output_files = sorted(glob.glob(os.path.join(rife_output_dir, "*")))
    for i, src in enumerate(output_files, start=1):
        ext = os.path.splitext(src)[1].lower()
        dst = os.path.join(rife_output_dir, f"{i:04d}.jpg")
        if src != dst:
            # Convert PNG to JPG if needed
            if ext == ".png":
                import cv2

                img = cv2.imread(src)
                if img is not None:
                    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    os.remove(src)
            else:
                os.rename(src, dst)

    # Replace original frames with interpolated ones
    # Remove original frames
    for f in glob.glob(os.path.join(temp_directory_path, "*.jpg")):
        os.remove(f)
    for f in glob.glob(os.path.join(temp_directory_path, "*.png")):
        os.remove(f)

    # Move interpolated frames to original directory
    for f in glob.glob(os.path.join(rife_output_dir, "*.jpg")):
        shutil.move(f, os.path.join(temp_directory_path, os.path.basename(f)))

    # Clean up RIFE output directory
    shutil.rmtree(rife_output_dir, ignore_errors=True)

    # Recount final frames in the temp directory
    final_count = _count_frames(temp_directory_path)
    _update_status(
        f"RIFE interpolation complete: {input_frame_count} -> {final_count} frames"
    )

    return final_count
