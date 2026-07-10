import importlib
import threading
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from types import ModuleType
from typing import Any

import cv2
from tqdm import tqdm

import modules
import modules.globals
from modules.processing_config_factory import build_config_from_globals

FRAME_PROCESSORS_MODULES: list[ModuleType] = []
_PROCESSORS_LOCK = threading.Lock()  # Protects FRAME_PROCESSORS_MODULES from concurrent access
FRAME_PROCESSORS_INTERFACE = ["pre_check", "pre_start", "process_frame", "process_image", "process_video"]


def _get_processors_snapshot() -> list[ModuleType]:
    """Return a thread-safe snapshot of FRAME_PROCESSORS_MODULES.

    Acquires the lock to ensure a consistent view of the list,
    returning a shallow copy to prevent external mutations.
    """
    with _PROCESSORS_LOCK:
        return list(FRAME_PROCESSORS_MODULES)


def load_frame_processor_module(frame_processor: str) -> Any:
    try:
        frame_processor_module = importlib.import_module(f"modules.processors.frame.{frame_processor}")
        for method_name in FRAME_PROCESSORS_INTERFACE:
            if not hasattr(frame_processor_module, method_name):
                raise ImportError(f"Frame processor '{frame_processor}' missing method: {method_name}")
    except ImportError as e:
        raise ImportError(f"Frame processor '{frame_processor}' could not be loaded: {e}") from e
    return frame_processor_module


def get_frame_processors_modules(frame_processors: list[str]) -> list[ModuleType]:
    global FRAME_PROCESSORS_MODULES

    with _PROCESSORS_LOCK:
        if not FRAME_PROCESSORS_MODULES:
            for frame_processor in frame_processors:
                frame_processor_module = load_frame_processor_module(frame_processor)
                FRAME_PROCESSORS_MODULES.append(frame_processor_module)

    set_frame_processors_modules_from_ui(frame_processors)
    return _get_processors_snapshot()


def set_frame_processors_modules_from_ui(frame_processors: list[str], config=None) -> None:
    global FRAME_PROCESSORS_MODULES

    config = config or build_config_from_globals()

    with _PROCESSORS_LOCK:
        current_processor_names = [proc.__name__.split(".")[-1] for proc in FRAME_PROCESSORS_MODULES]

        for frame_processor, state in config.fp_ui.items():
            if state and frame_processor not in current_processor_names:
                try:
                    frame_processor_module = load_frame_processor_module(frame_processor)
                    FRAME_PROCESSORS_MODULES.append(frame_processor_module)
                    if frame_processor not in modules.globals.frame_processors:
                        modules.globals.frame_processors.append(frame_processor)
                    # Trigger model download in the background so the UI
                    # stays responsive.  pre_check() is normally only called
                    # at startup for initially-enabled processors.
                    threading.Thread(
                        target=frame_processor_module.pre_check,
                        daemon=True,
                        name=f"dl-{frame_processor}",
                    ).start()
                except SystemExit:
                    print(f"Warning: Failed to load frame processor {frame_processor} requested by UI state.")
                except Exception as e:
                    print(f"Warning: Error loading frame processor {frame_processor} requested by UI state: {e}")

            elif not state and frame_processor in current_processor_names:
                try:
                    module_to_remove = next(
                        (mod for mod in FRAME_PROCESSORS_MODULES if mod.__name__.endswith(f".{frame_processor}")), None
                    )
                    if module_to_remove:
                        FRAME_PROCESSORS_MODULES.remove(module_to_remove)
                    if frame_processor in modules.globals.frame_processors:
                        modules.globals.frame_processors.remove(frame_processor)
                except Exception as e:
                    print(f"Warning: Error removing frame processor {frame_processor}: {e}")


def multi_process_frame(
    source_path: str,
    temp_frame_paths: list[str],
    process_frames: Callable[[str, list[str], Any], None],
    progress: Any = None,
    config=None,
) -> None:
    """Process video frames in parallel using ProcessPoolExecutor.

    Uses separate processes for video batch mode to bypass the GIL and fully
    utilise multiple CPU cores. Each worker process loads its own ONNX model
    (~2-5s startup overhead per worker), which is amortised over the batch of
    frames assigned to each worker.

    Frames are grouped into batches (one per worker) so each process loads
    the ONNX model once and processes many frames, rather than paying the
    model-loading cost per individual frame.

    For live/webcam mode, use multi_process_frame_live() instead which uses
    ThreadPoolExecutor to avoid per-process model loading latency.
    """
    config = config or build_config_from_globals()
    max_workers = config.execution_threads

    # Split frame paths into batches — one batch per worker process.
    # This amortises the per-process ONNX model loading (~2-5s) across
    # many frames instead of paying it once per frame.
    batches: list[list[str]] = [[] for _ in range(max_workers)]
    for i, path in enumerate(temp_frame_paths):
        batches[i % max_workers].append(path)
    # Remove empty batches (when fewer frames than workers)
    batches = [b for b in batches if b]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_frames, source_path, batch, None): batch for batch in batches}
        for future in as_completed(futures):
            batch = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing batch of {len(batch)} frames: {e}")
            if progress:
                progress.update(len(batch))


def multi_process_frame_live(
    source_path: str,
    temp_frame_paths: list[str],
    process_frames: Callable[[str, list[str], Any], None],
    progress: Any = None,
    config=None,
) -> None:
    """Process frames in parallel using ThreadPoolExecutor for live mode.

    ThreadPoolExecutor avoids the ~2-5s per-worker model loading overhead of
    ProcessPoolExecutor, which is critical for live/webcam mode where latency
    matters. ONNX Runtime releases the GIL during inference, so threads still
    get reasonable parallelism on the dominant cost (model inference).
    """
    config = config or build_config_from_globals()
    max_workers = config.execution_threads

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_frames, source_path, [path], progress): path for path in temp_frame_paths}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing frame {futures[future]}: {e}")


def process_frames_io(
    temp_frame_paths: list[str],
    process_fn: Callable,
    progress: Any = None,
    jpeg_quality: int = 95,
) -> None:
    """Read/process/write loop shared by frame processors.

    ``process_fn`` receives a single frame (numpy array) and must return
    the processed frame.  Frames that cannot be read are skipped.
    """
    for path in temp_frame_paths:
        frame = cv2.imread(path)
        if frame is None:
            if progress:
                progress.update(1)
            continue
        result = process_fn(frame)
        if result is None:
            result = frame
        cv2.imwrite(path, result, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if progress:
            progress.update(1)


def process_video(
    source_path: str, frame_paths: list[str], process_frames: Callable[[str, list[str], Any], None], config=None
) -> None:
    config = config or build_config_from_globals()
    progress_bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    total = len(frame_paths)
    with tqdm(
        total=total, desc="Processing", unit="frame", dynamic_ncols=True, bar_format=progress_bar_format
    ) as progress:
        progress.set_postfix(
            {
                "execution_providers": config.execution_providers,
                "execution_threads": config.execution_threads,
                "max_memory": config.max_memory,
            }
        )
        multi_process_frame(source_path, frame_paths, process_frames, progress, config)
