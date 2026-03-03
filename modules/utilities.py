import glob
import hashlib
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import urllib
import urllib.request
from pathlib import Path
from typing import List, Any, Optional
from tqdm import tqdm

import modules.globals
from modules.processing_config import ProcessingConfig
from modules.processing_config_factory import build_config_from_globals

TEMP_FILE = "temp.mp4"
TEMP_DIRECTORY = "temp"

# Module-level download progress callback.
# Signature: (filename: str, downloaded: int, total: int) -> None
_download_progress_callback = None


def set_download_progress_callback(cb) -> None:
    """Register a callback to receive download progress updates."""
    global _download_progress_callback
    _download_progress_callback = cb


def clear_download_progress_callback() -> None:
    """Remove the download progress callback."""
    global _download_progress_callback
    _download_progress_callback = None


def _compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file, reading in 1 MiB chunks to handle large models."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_path_for_subprocess(path: str) -> None:
    """Raise ValueError if a file path basename starts with '-', preventing argument injection."""
    if os.path.basename(path).startswith("-"):
        raise ValueError(
            f"Unsafe file path rejected: basename of {path!r} starts with '-'. "
            "Rename the file to use it with this application."
        )


def run_ffmpeg(args: List[str], config: Optional[ProcessingConfig] = None) -> bool:
    """Run ffmpeg with hardware acceleration and optimized settings."""
    if config is None:
        config = build_config_from_globals()
    commands = [
        "ffmpeg",
        "-hide_banner",
        "-hwaccel", "auto",  # Auto-detect hardware acceleration
        "-hwaccel_output_format", "auto",  # Use hardware format when possible
        "-threads", str(config.execution_threads or 0),  # 0 = auto-detect optimal thread count
        "-loglevel", config.log_level,
    ]
    commands.extend(args)
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except Exception as e:
        print(f"run_ffmpeg: command failed: {e}")
    return False


def detect_fps(target_path: str) -> float:
    _validate_path_for_subprocess(target_path)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        target_path,
    ]
    output = subprocess.check_output(command).decode().strip().split("/")
    try:
        numerator, denominator = map(int, output)
        return numerator / denominator
    except Exception as e:
        print(f"detect_fps: could not parse frame rate from {target_path!r}: {e}, defaulting to 30.0")
    return 30.0


def extract_frames(target_path: str, config: Optional[ProcessingConfig] = None) -> None:
    """Extract frames with hardware acceleration and optimized settings."""
    _validate_path_for_subprocess(target_path)
    if config is None:
        config = build_config_from_globals()
    temp_directory_path = get_temp_directory_path(target_path)

    if config.use_png_frames:
        frame_pattern = os.path.join(temp_directory_path, "%04d.png")
        extra_args: list = []
    else:
        frame_pattern = os.path.join(temp_directory_path, "%04d.jpg")
        extra_args = ["-qscale:v", "2"]  # JPEG quality ~95% (scale 2-31, lower=better)

    # Use hardware-accelerated decoding and optimized pixel format
    run_ffmpeg(
        [
            "-i", target_path,
            "-vf", "format=rgb24",  # Use video filter for format conversion (faster)
            "-vsync", "0",  # Prevent frame duplication
            "-frame_pts", "1",  # Preserve frame timing
            frame_pattern,
            *extra_args,
        ],
        config=config,
    )


_NVIDIA_ENCODERS = {'libx264': 'h264_nvenc', 'libx265': 'hevc_nvenc'}
_AMD_ENCODERS = {'libx264': 'h264_amf', 'libx265': 'hevc_amf'}
_HW_ENCODERS = set(_NVIDIA_ENCODERS.values()) | set(_AMD_ENCODERS.values())


def _build_encoder_args(software_encoder: str, quality: int | None, providers: list) -> tuple[str, list]:
    """Return (encoder_name, extra_ffmpeg_flags) for the given hardware context.

    Pure function — reads only its arguments, has no side effects.
    """
    if quality is None:
        quality = 18
    if 'CUDAExecutionProvider' in providers:
        encoder = _NVIDIA_ENCODERS.get(software_encoder, software_encoder)
        if encoder in ('h264_nvenc', 'hevc_nvenc'):
            options = [
                "-preset", "p7",
                "-tune", "hq",
                "-rc", "vbr",
                "-cq", str(quality),
                "-b:v", "0",
                "-multipass", "fullres",
            ]
            return encoder, options
    elif 'DmlExecutionProvider' in providers:
        encoder = _AMD_ENCODERS.get(software_encoder, software_encoder)
        if encoder in ('h264_amf', 'hevc_amf'):
            options = [
                "-quality", "quality",
                "-rc", "vbr_latency",
                "-qp_i", str(quality),
                "-qp_p", str(quality),
            ]
            return encoder, options

    # CPU encoding
    if software_encoder == 'libx264':
        return software_encoder, ["-preset", "medium", "-crf", str(quality), "-tune", "film"]
    if software_encoder == 'libx265':
        return software_encoder, ["-preset", "medium", "-crf", str(quality), "-x265-params", "log-level=error"]
    if software_encoder == 'libvpx-vp9':
        return software_encoder, ["-crf", str(quality), "-b:v", "0", "-cpu-used", "2"]
    return software_encoder, []


def _build_video_ffmpeg_args(
    fps: float,
    input_pattern: str,
    encoder: str,
    encoder_options: list,
    output_path: str,
) -> list:
    """Build the full ffmpeg argument list for video encoding.

    Pure function — no side effects.
    """
    return [
        "-r", str(fps),
        "-i", input_pattern,
        "-c:v", encoder,
        *encoder_options,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-vf", "colorspace=bt709:iall=bt601-6-625:fast=1",
        "-y",
        output_path,
    ]


def create_video(target_path: str, fps: float = 30.0,
                 config: Optional[ProcessingConfig] = None) -> None:
    """Create video with hardware-accelerated encoding and optimized settings."""
    _validate_path_for_subprocess(target_path)
    if config is None:
        config = build_config_from_globals()
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    ext = "png" if config.use_png_frames else "jpg"
    input_pattern = os.path.join(temp_directory_path, f"%04d.{ext}")

    encoder, encoder_options = _build_encoder_args(
        config.video_encoder,
        config.video_quality,
        config.execution_providers,
    )
    ffmpeg_args = _build_video_ffmpeg_args(fps, input_pattern, encoder, encoder_options, temp_output_path)

    success = run_ffmpeg(ffmpeg_args, config=config)

    if not success and encoder in _HW_ENCODERS:
        print(f"Hardware encoding with {encoder} failed, falling back to software encoding...")
        fallback_encoder = 'libx264' if 'h264' in encoder else 'libx265'
        _, fallback_options = _build_encoder_args(fallback_encoder, config.video_quality, [])
        fallback_args = _build_video_ffmpeg_args(fps, input_pattern, fallback_encoder, fallback_options, temp_output_path)
        run_ffmpeg(fallback_args, config=config)


def restore_audio(target_path: str, output_path: str) -> None:
    _validate_path_for_subprocess(target_path)
    _validate_path_for_subprocess(output_path)
    temp_output_path = get_temp_output_path(target_path)
    done = run_ffmpeg(
        [
            "-i",
            temp_output_path,
            "-i",
            target_path,
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-y",
            output_path,
        ]
    )
    if not done:
        move_temp(target_path, output_path)


def get_temp_frame_paths(target_path: str,
                         config: Optional[ProcessingConfig] = None) -> List[str]:
    if config is None:
        config = build_config_from_globals()
    temp_directory_path = get_temp_directory_path(target_path)
    ext = "png" if config.use_png_frames else "jpg"
    return glob.glob(os.path.join(glob.escape(temp_directory_path), f"*.{ext}"))


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_FILE)


def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Any:
    if source_path:
        _validate_path_for_subprocess(source_path)
    if target_path:
        _validate_path_for_subprocess(target_path)
    if source_path and target_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if output_path and os.path.isdir(output_path):
            return os.path.join(
                output_path, source_name + "-" + target_name + target_extension
            )
    return output_path


def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)


def clean_temp(target_path: str, config: Optional[ProcessingConfig] = None) -> None:
    if config is None:
        config = build_config_from_globals()
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not config.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)


def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith("image/"))
    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith("video/"))
    return False


def conditional_download(download_directory_path: str, urls: List[str], expected_checksums: dict | None = None) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        filename = os.path.basename(url)
        download_file_path = os.path.join(download_directory_path, filename)
        if not os.path.exists(download_file_path):
            ssl_context = ssl.create_default_context()
            request = urllib.request.urlopen(url, context=ssl_context)  # type: ignore[attr-defined]
            total = int(request.headers.get("Content-Length", 0))
            downloaded = [0]  # mutable for closure access

            if _download_progress_callback:
                _download_progress_callback(filename, 0, total)

            def _reporthook(count, block_size, total_size):
                progress.update(block_size)
                downloaded[0] += block_size
                if _download_progress_callback:
                    _download_progress_callback(filename, min(downloaded[0], total_size), total_size)

            with tqdm(
                total=total,
                desc=f"Downloading {filename}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                urllib.request.urlretrieve(url, download_file_path, reporthook=_reporthook)  # type: ignore[attr-defined]

            if _download_progress_callback:
                _download_progress_callback(filename, total, total)

            # Verify checksum if one was registered for this file
            if expected_checksums and filename in expected_checksums:
                expected = expected_checksums[filename]
                actual = _compute_sha256(download_file_path)
                if actual != expected:
                    os.remove(download_file_path)
                    raise ValueError(
                        f"Checksum mismatch for {filename!r}: "
                        f"expected {expected!r}, got {actual!r}. "
                        f"File deleted — please retry the download."
                    )
                print(f"conditional_download: checksum verified for {filename!r}")


def download_model_if_needed(
    model_file: str,
    model_urls: list[str],
    processor_name: str,
    models_dir: str | None = None,
    on_status=None,
) -> bool:
    """Download a model if not present. Returns True if model is available.

    *on_status* is an optional ``Callable[[str, str], None]`` that receives
    ``(message, processor_name)`` status updates.  When omitted the function
    falls back to importing ``modules.core.update_status`` (legacy behaviour).
    """
    if models_dir is None:
        from modules.paths import MODELS_DIR
        models_dir = MODELS_DIR
    model_path = os.path.join(models_dir, model_file)

    def _emit(msg: str) -> None:
        if on_status is not None:
            on_status(msg, processor_name)
        else:
            from modules.core import update_status
            update_status(msg, processor_name)

    if not os.path.exists(model_path):
        _emit(f"Downloading {model_file}...")
    conditional_download(models_dir, model_urls)
    if not os.path.exists(model_path):
        _emit(f"Model not found at {model_path}. Download may have failed.")
        return False
    return True


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))
