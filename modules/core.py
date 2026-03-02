import os
import sys
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import threading
import warnings
from typing import List, Optional
import platform
import signal
import shutil
import argparse
import onnxruntime

import modules.globals
import modules.metadata
from modules.processing_config import ProcessingConfig
from modules.processing_config_factory import build_config_from_globals
from modules.status_bus import BUS as _STATUS_BUS
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path, set_download_progress_callback
from modules.blas_check import check_apple_silicon_blas

# Lazy-loaded heavy imports — deferred until actually needed to save 3-5s startup.
torch = None  # loaded on demand by _ensure_torch()
HAS_TORCH = None  # tri-state: None = not checked, True/False after check

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')


def _ensure_torch() -> bool:
    """Import torch on first call and cache the result."""
    global torch, HAS_TORCH
    if HAS_TORCH is None:
        try:
            import torch as _torch
            torch = _torch
            HAS_TORCH = True
            warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
        except ImportError:
            HAS_TORCH = False
    return HAS_TORCH


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser()
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='pipeline of frame processors', dest='frame_processor', default=['face_swapper'], choices=['face_swapper', 'face_enhancer', 'face_enhancer_gpen256', 'face_enhancer_gpen512', 'face_enhancer_codeformer'], nargs='+')
    program.add_argument('--png-frames', help='use lossless PNG for intermediate frames (larger files, no artifacts)', dest='use_png_frames', action='store_true', default=False)
    program.add_argument('--keep-fps', help='keep original fps', dest='keep_fps', action='store_true', default=False)
    program.add_argument('--keep-audio', help='keep original audio', dest='keep_audio', action='store_true', default=True)
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true', default=False)
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true', default=False)
    program.add_argument('--nsfw-filter', help='filter the NSFW image or video', dest='nsfw_filter', action='store_true', default=False)
    program.add_argument('--map-faces', help='map source target faces', dest='map_faces', action='store_true', default=False)
    program.add_argument('--mouth-mask', help='mask the mouth region', dest='mouth_mask', action='store_true', default=False)
    program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int, default=18, choices=range(52), metavar='[0-51]')
    program.add_argument('-l', '--lang', help='Ui language', default="en")
    program.add_argument('--live-mirror', help='The live camera display as you see it in the front-facing camera frame', dest='live_mirror', action='store_true', default=False)
    program.add_argument('--live-resizable', help='The live camera frame is resizable', dest='live_resizable', action='store_true', default=False)
    program.add_argument('--virtual-cam', help='output to virtual camera device', dest='virtual_cam', action='store_true', default=False)
    program.add_argument('--rife', help='enable RIFE frame interpolation for video output', dest='rife_enabled', action='store_true', default=False)
    program.add_argument('--rife-model', help='RIFE model to use', dest='rife_model', default='rife-v4.25-lite', choices=['rife-v4.25', 'rife-v4.25-lite'])
    program.add_argument('--rife-multiplier', help='RIFE frame rate multiplier (2=double, 4=quadruple)', dest='rife_multiplier', type=int, default=2, choices=[2, 4])
    program.add_argument('--half-rate', help='enable half-rate face processing with RIFE interpolation for live mode', dest='half_rate_processing', action='store_true', default=False)
    program.add_argument('--keyframe-interval', help='process every Nth frame in half-rate mode (2-10)', dest='keyframe_interval', type=int, default=2, choices=range(2, 11))
    program.add_argument('--live-enhance-size', help='face alignment resolution for enhancement in live mode; smaller values reduce warp/paste cost (default: 256)', dest='live_enhance_size', type=int, default=256, choices=[128, 192, 256, 384, 512])
    program.add_argument('--landmark-smoothing', help='enable EMA smoothing of face landmarks and bounding boxes in live mode to reduce jitter', dest='landmark_smoothing', action='store_true', default=False)
    program.add_argument('--landmark-smoothing-alpha', help='EMA alpha for landmark smoothing: weight for current frame (0.0=full smoothing, 1.0=no smoothing, default: 0.7)', dest='landmark_smoothing_alpha', type=float, default=0.7)
    program.add_argument('--face-swap-model', help='face swap model variant to use (default: inswapper)', dest='face_swap_model', default='inswapper', choices=['inswapper', 'ghost_256_v1', 'ghost_256_v2', 'ghost_256_v3'])
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='execution provider', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{modules.metadata.name} {modules.metadata.version}')

    # register deprecated args
    program.add_argument('-f', '--face', help=argparse.SUPPRESS, dest='source_path_deprecated')
    program.add_argument('--cpu-cores', help=argparse.SUPPRESS, dest='cpu_cores_deprecated', type=int)
    program.add_argument('--gpu-vendor', help=argparse.SUPPRESS, dest='gpu_vendor_deprecated')
    program.add_argument('--gpu-threads', help=argparse.SUPPRESS, dest='gpu_threads_deprecated', type=int)

    args = program.parse_args()

    modules.globals.source_path = args.source_path
    modules.globals.target_path = args.target_path
    modules.globals.output_path = normalize_output_path(modules.globals.source_path, modules.globals.target_path, args.output_path)
    modules.globals.frame_processors = args.frame_processor
    modules.globals.headless = args.source_path or args.target_path or args.output_path
    modules.globals.use_png_frames = args.use_png_frames
    modules.globals.keep_fps = args.keep_fps
    modules.globals.keep_audio = args.keep_audio
    modules.globals.keep_frames = args.keep_frames
    modules.globals.many_faces = args.many_faces
    modules.globals.mouth_mask = args.mouth_mask
    modules.globals.nsfw_filter = args.nsfw_filter
    modules.globals.map_faces = args.map_faces
    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.live_mirror = args.live_mirror
    modules.globals.live_resizable = args.live_resizable
    modules.globals.virtual_cam = args.virtual_cam
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider)
    modules.globals.execution_threads = args.execution_threads
    modules.globals.lang = args.lang
    modules.globals.rife_enabled = args.rife_enabled
    modules.globals.rife_model = args.rife_model
    modules.globals.rife_multiplier = args.rife_multiplier
    modules.globals.half_rate_processing = args.half_rate_processing
    modules.globals.keyframe_interval = args.keyframe_interval
    modules.globals.live_enhance_size = args.live_enhance_size
    modules.globals.face_swap_model = args.face_swap_model
    modules.globals.landmark_smoothing = args.landmark_smoothing
    modules.globals.landmark_smoothing_alpha = args.landmark_smoothing_alpha

    #for ENHANCER tumblers:
    for enhancer_key in ('face_enhancer', 'face_enhancer_gpen256', 'face_enhancer_gpen512', 'face_enhancer_codeformer'):
        modules.globals.fp_ui[enhancer_key] = enhancer_key in args.frame_processor

    # translate deprecated args
    if args.source_path_deprecated:
        print('\033[33mArgument -f and --face are deprecated. Use -s and --source instead.\033[0m')
        modules.globals.source_path = args.source_path_deprecated
        modules.globals.output_path = normalize_output_path(args.source_path_deprecated, modules.globals.target_path, args.output_path)
    if args.cpu_cores_deprecated:
        print('\033[33mArgument --cpu-cores is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.cpu_cores_deprecated
    if args.gpu_vendor_deprecated == 'apple':
        print('\033[33mArgument --gpu-vendor apple is deprecated. Use --execution-provider coreml instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['coreml'])
    if args.gpu_vendor_deprecated == 'nvidia':
        print('\033[33mArgument --gpu-vendor nvidia is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['cuda'])
    if args.gpu_vendor_deprecated == 'amd':
        print('\033[33mArgument --gpu-vendor amd is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['rocm'])
    if args.gpu_threads_deprecated:
        print('\033[33mArgument --gpu-threads is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.gpu_threads_deprecated


# Cache available providers at import time — querying the ONNX Runtime
# registry is not free and the result never changes within a process.
_AVAILABLE_PROVIDERS: List[str] = onnxruntime.get_available_providers()


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(_AVAILABLE_PROVIDERS, encode_execution_providers(_AVAILABLE_PROVIDERS))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(_AVAILABLE_PROVIDERS)


def suggest_execution_threads() -> int:
    """Suggest optimal thread count based on hardware and execution provider."""
    import os

    # Get CPU count
    cpu_count = os.cpu_count() or 4

    if 'DmlExecutionProvider' in modules.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in modules.globals.execution_providers:
        return 1
    if 'TensorrtExecutionProvider' in modules.globals.execution_providers:
        # TensorRT handles GPU parallelism internally; a single CPU thread
        # avoids contention on the TRT execution context.
        return 1
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        # For CUDA, use more threads for parallel frame processing
        return min(cpu_count, 16)

    # For CPU execution, use most cores but leave some for system
    return max(4, min(cpu_count - 2, 16))


def limit_resources(config: Optional[ProcessingConfig] = None) -> None:
    if config is None:
        config = build_config_from_globals()
    # prevent tensorflow memory leak (lazy import — only needed for GPU memory config)
    try:
        import tensorflow
        gpus = tensorflow.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
    except ImportError:
        pass
    # limit memory usage
    if config.max_memory:
        memory = config.max_memory * 1024 ** 3
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        elif platform.system().lower() == 'linux':
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources(config: Optional[ProcessingConfig] = None) -> None:
    if config is None:
        config = build_config_from_globals()
    if (
        'CUDAExecutionProvider' in config.execution_providers
        or 'TensorrtExecutionProvider' in config.execution_providers
    ) and _ensure_torch():
        torch.cuda.empty_cache()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    # Check NumPy BLAS configuration on Apple Silicon (informational, not fatal)
    check_apple_silicon_blas()
    # Warn about TensorRT first-run engine compilation delay
    if 'TensorrtExecutionProvider' in modules.globals.execution_providers:
        from modules.tensorrt_cache import has_cached_engines
        if not has_cached_engines():
            update_status(
                'TensorRT selected: first-run engine compilation may take 30-120s per model. '
                'Engines will be cached in models/trt_cache/ for instant startup on future runs.'
            )
    return True


def update_status(message: str, scope: str = 'DLC.CORE') -> None:
    print(f'[{scope}] {message}')
    _STATUS_BUS.publish(message, scope)

def start(config: Optional[ProcessingConfig] = None) -> None:
    """Start processing with performance monitoring."""
    import time

    if config is None:
        config = build_config_from_globals()

    start_time = time.time()

    for frame_processor in get_frame_processors_modules(config.frame_processors):
        if not frame_processor.pre_start():
            return
    update_status('Processing...')

    # process image to image
    if has_image_extension(config.target_path):
        if config.nsfw_filter:
            import modules.ui as ui
            if ui.check_and_ignore_nsfw(config.target_path, destroy):
                return
        try:
            shutil.copy2(config.target_path, config.output_path)
        except Exception as e:
            print("Error copying file:", str(e))
        for frame_processor in get_frame_processors_modules(config.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(config.source_path, config.output_path, config.output_path)
            release_resources(config)
        if is_image(config.target_path):
            elapsed = time.time() - start_time
            update_status(f'Processing to image succeed! (Time: {elapsed:.2f}s)')
        else:
            update_status('Processing to image failed!')
        return

    # process image to videos
    if config.nsfw_filter:
        import modules.ui as ui
        if ui.check_and_ignore_nsfw(config.target_path, destroy):
            return

    extraction_start = time.time()
    if not config.map_faces:
        update_status('Creating temp resources...')
        create_temp(config.target_path)
        update_status('Extracting frames...')
        extract_frames(config.target_path, config=config)
    extraction_time = time.time() - extraction_start
    update_status(f'Frame extraction completed in {extraction_time:.2f}s')

    temp_frame_paths = get_temp_frame_paths(config.target_path, config=config)
    total_frames = len(temp_frame_paths)
    update_status(f'Processing {total_frames} frames with {config.execution_threads} threads...')

    processing_start = time.time()
    for frame_processor in get_frame_processors_modules(config.frame_processors):
        update_status('Progressing...', frame_processor.NAME)
        frame_processor.process_video(config.source_path, temp_frame_paths)
        release_resources(config)
    processing_time = time.time() - processing_start
    fps_processing = total_frames / processing_time if processing_time > 0 else 0
    update_status(f'Frame processing completed in {processing_time:.2f}s ({fps_processing:.2f} fps)')

    # RIFE frame interpolation (if enabled)
    if config.rife_enabled:
        from modules.rife_interpolation import interpolate_frames
        rife_start = time.time()
        temp_directory_path = os.path.dirname(temp_frame_paths[0]) if temp_frame_paths else None
        if temp_directory_path:
            new_count = interpolate_frames(temp_directory_path)
            rife_time = time.time() - rife_start
            if new_count:
                update_status(f'RIFE interpolation completed in {rife_time:.2f}s ({new_count} frames)')
                # Refresh frame paths after interpolation added new frames
                temp_frame_paths = get_temp_frame_paths(config.target_path, config=config)
            else:
                update_status(f'RIFE interpolation skipped or failed ({rife_time:.2f}s)')

    # handles fps
    encoding_start = time.time()
    rife_multiplier = config.rife_multiplier if config.rife_enabled else 1
    if config.keep_fps:
        update_status('Detecting fps...')
        fps = detect_fps(config.target_path) * rife_multiplier
        update_status(f'Creating video with {fps} fps...')
        create_video(config.target_path, fps, config=config)
    else:
        adjusted_fps = 30.0 * rife_multiplier
        update_status(f'Creating video with {adjusted_fps:.1f} fps...')
        create_video(config.target_path, adjusted_fps, config=config)
    encoding_time = time.time() - encoding_start
    update_status(f'Video encoding completed in {encoding_time:.2f}s')

    # handle audio
    if config.keep_audio:
        if config.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(config.target_path, config.output_path)
    else:
        move_temp(config.target_path, config.output_path)

    # clean and validate
    clean_temp(config.target_path, config=config)

    total_time = time.time() - start_time
    if is_video(config.target_path):
        update_status(f'Processing to video succeed! Total time: {total_time:.2f}s')
    else:
        update_status('Processing to video failed!')


def destroy(to_quit=True) -> None:
    if modules.globals.target_path:
        clean_temp(modules.globals.target_path)
    if to_quit:
        try:
            import modules.ui as ui
            if ui.ROOT is not None:
                ui.ROOT.quit()
        except Exception:
            pass
        raise SystemExit(0)


def _run_processor_pre_checks(config: Optional[ProcessingConfig] = None) -> None:
    """Download missing models in a background thread (GUI mode only).

    Runs each frame processor's pre_check() sequentially so tqdm progress bars
    don't interleave. Status messages are posted to the UI via update_status().
    RIFE pre_check runs last since it depends on the frame processors being ready.
    """
    if config is None:
        config = build_config_from_globals()
    for frame_processor in get_frame_processors_modules(config.frame_processors):
        frame_processor.pre_check()
    if config.rife_enabled:
        from modules.rife_interpolation import pre_check as rife_pre_check
        rife_pre_check()


def run() -> None:
    from modules.processing_config_factory import build_config_from_cli_args
    parse_args()
    if not pre_check():
        return
    # Build a config snapshot from the globals that parse_args() just wrote.
    # Headless mode uses this config exclusively; GUI mode still mutates globals
    # via widget callbacks, so it re-snapshots at each processing start.
    config = build_config_from_globals()
    limit_resources(config)
    if config.headless:
        # Headless: run pre_checks synchronously before processing starts.
        for frame_processor in get_frame_processors_modules(config.frame_processors):
            if not frame_processor.pre_check():
                return
        if config.rife_enabled:
            from modules.rife_interpolation import pre_check as rife_pre_check
            if not rife_pre_check():
                return
        start(config)
    else:
        # GUI mode: start the UI immediately so the webcam preview is responsive,
        # then download any missing models in the background.  Each processor's
        # process_frame/swap_face already returns the original frame when its
        # model is not yet loaded, so the live feed stays smooth during download.
        import modules.ui as ui
        _STATUS_BUS.subscribe(lambda msg, _: ui.update_status(msg))
        set_download_progress_callback(ui.download_progress_callback)
        threading.Thread(target=_run_processor_pre_checks, daemon=True, name="model-downloader").start()
        window = ui.init(start, destroy, modules.globals.lang)
        window.mainloop()
