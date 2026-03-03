import cv2
import time
import queue
import threading
from typing import Optional
from PIL import Image, ImageDraw
import customtkinter as ctk

import modules.globals
from modules import virtual_cam
from modules.processing_config import ProcessingConfig
from modules.processing_config_factory import build_config_from_globals
from modules.gpu_processing import gpu_cvt_color, gpu_flip
from modules.face_analyser import (
    get_one_face, get_many_faces, set_det_size,
    detect_faces_for_webcam, faces_are_similar, FaceAnalyser, LandmarkSmoother,
)
from modules.processors.frame.core import get_frame_processors_modules
from modules.rife_interpolation import has_native_binding, interpolate_frame_pair, cleanup_rife
from modules.single_slot_worker import single_slot_worker_loop
from modules.video_capture import VideoCapturer


# DETECT_EVERY_N is kept for backward-compatibility with any external imports
# but is no longer used by the processing thread — detection now runs in its
# own dedicated thread.
DETECT_EVERY_N = 2

# Module-level session tracking — prevents concurrent camera sessions.
# _active_stop_event: stop event of the currently running session (None if none).
# _session_ready: set when no session is running (or cleanup has finished).
_active_stop_event: Optional[threading.Event] = None
_session_ready: threading.Event = threading.Event()
_session_ready.set()  # No session running initially

# Enhancer processor names for skip-frame logic
_ENHANCER_NAMES = frozenset({
    "DLC.FACE-ENHANCER",
    "DLC.FACE-ENHANCER-GPEN256",
    "DLC.FACE-ENHANCER-GPEN512",
})

# Map from processor NAME to fp_ui toggle key
_ENHANCER_UI_KEYS = {
    "DLC.FACE-ENHANCER": "face_enhancer",
    "DLC.FACE-ENHANCER-GPEN256": "face_enhancer_gpen256",
    "DLC.FACE-ENHANCER-GPEN512": "face_enhancer_gpen512",
}


def _is_enhancer_enabled(processor, fp_ui: Optional[dict] = None) -> bool:
    """Check if an enhancer processor is toggled on in the UI."""
    if fp_ui is None:
        fp_ui = modules.globals.fp_ui
    key = _ENHANCER_UI_KEYS.get(processor.NAME)
    return key is not None and fp_ui.get(key, False)


def stop_active_session(timeout: float = 0.5) -> None:
    """Signal the active webcam session to stop and wait for threads to finish.

    Call this before Python's shutdown sequence to prevent SIGSEGV crashes
    where daemon threads are still running MLX/ONNX inference when native
    library resources are freed during shutdown.  Safe to call with no
    active session.
    """
    if _active_stop_event is not None and not _active_stop_event.is_set():
        _active_stop_event.set()
    # Give threads up to `timeout` seconds to finish their current inference
    # cycle and exit their loops before Python tears down native resources.
    # _session_ready is cleared at session start and set again after cleanup;
    # if cleanup doesn't run (e.g., ROOT.quit() stopped the display loop),
    # the wait just times out — which is still enough grace time.
    _session_ready.wait(timeout=timeout)


def _capture_thread_func(cap, capture_queue, stop_event):
    """Capture thread: reads frames from camera and puts them into the queue.
    Drops frames when the queue is full to avoid backpressure on the camera."""
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break
        try:
            capture_queue.put_nowait(frame)
        except queue.Full:
            # Drop the oldest frame and enqueue the new one
            try:
                capture_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                capture_queue.put_nowait(frame)
            except queue.Full:
                pass


def _detection_thread_func(latest_frame_holder, detection_result, detection_lock, stop_event,
                            config: Optional[ProcessingConfig] = None,
                            smoother: Optional[LandmarkSmoother] = None):
    """Detection thread (producer): continuously reads the most recently
    captured raw frame and runs face detection on it, storing results in
    *detection_result* under *detection_lock*.

    latest_frame_holder is a one-element list [frame | None] written by the
    processing thread so the detection thread always works on the newest frame
    without queuing overhead.  The detection thread never touches Tkinter
    widgets — all UI updates go through ROOT.after() in the display loop.

    When *smoother* is provided and ``modules.globals.landmark_smoothing`` is
    enabled, EMA smoothing is applied to the detected face bounding boxes and
    keypoints before the result is stored.  The smoother's ``alpha`` is kept
    in sync with ``modules.globals.landmark_smoothing_alpha`` on each frame
    so the user can tune it live without restarting the session.
    """
    if config is None:
        config = build_config_from_globals()
    while not stop_event.is_set():
        with detection_lock:
            frame = latest_frame_holder[0]

        if frame is None:
            time.sleep(0.005)
            continue

        result = detect_faces_for_webcam(frame, many_faces=modules.globals.many_faces)

        # EMA landmark smoothing (live mode only; zero cost when disabled)
        if smoother is not None and modules.globals.landmark_smoothing:
            smoother.alpha = modules.globals.landmark_smoothing_alpha
            if result['many_faces']:
                smoother.smooth(result['many_faces'])
            elif result['target_face'] is not None:
                smoothed = smoother.smooth([result['target_face']])
                result['target_face'] = smoothed[0] if smoothed else result['target_face']
            else:
                smoother.reset()
        elif smoother is not None:
            # Smoothing was just disabled — clear stale state so the next enable
            # starts fresh rather than blending against old positions.
            smoother.reset()

        with detection_lock:
            detection_result['target_face'] = result['target_face']
            detection_result['many_faces'] = result['many_faces']


def _swap_process_fn(inp):
    """Process a single swap request (called by single_slot_worker_loop)."""
    frame = inp['frame']
    processor = inp['processor']
    # Build config once per request — use pre-built config from inp when available to
    # avoid reconstructing the ~80-field ProcessingConfig dataclass on every frame.
    config = inp.get('config') or build_config_from_globals()

    if inp['map_faces']:
        frame = processor.process_frame_v2(frame, config=config)
    else:
        source_face = inp['source_face']
        many_faces_list = inp['many_faces']
        target_face = inp['target_face']

        swapped_bboxes = []
        if many_faces_list:
            result = frame if config.opacity >= 1.0 else frame.copy()
            for t_face in many_faces_list:
                result = processor.swap_face(source_face, t_face, result, config=config)
                if hasattr(t_face, 'bbox') and t_face.bbox is not None:
                    swapped_bboxes.append(t_face.bbox.astype(int))
            frame = result
        elif target_face is not None:
            frame = processor.swap_face(source_face, target_face, frame, config=config)
            if hasattr(target_face, 'bbox') and target_face.bbox is not None:
                swapped_bboxes.append(target_face.bbox.astype(int))

        frame = processor.apply_post_processing(frame, swapped_bboxes, config=config)

    return {'frame': frame, 'seq': inp['seq']}


def _swap_thread_func(swap_input, swap_output, swap_lock, stop_event):
    """Swap thread: runs face swap ONNX inference asynchronously."""
    single_slot_worker_loop(swap_input, swap_output, swap_lock, stop_event,
                            _swap_process_fn)


def _enhancement_process_fn(inp):
    """Process a single enhancement request (called by single_slot_worker_loop)."""
    processor = inp['processor']
    frame = inp['frame']
    faces = inp['faces']
    map_faces = inp['map_faces']

    if map_faces:
        enhanced = processor.process_frame_v2(frame)
    else:
        enhanced = processor.process_frame(None, frame, faces=faces)

    return {'frame': enhanced, 'seq': inp['seq']}


def _enhancement_thread_func(enhancement_input, enhancement_output,
                              enhancement_lock, stop_event):
    """Enhancement thread: runs face enhancement (GFPGAN/GPEN) asynchronously."""
    single_slot_worker_loop(enhancement_input, enhancement_output,
                            enhancement_lock, stop_event,
                            _enhancement_process_fn)


def _processing_thread_func(capture_queue, processed_queue, stop_event,
                             latest_frame_holder, detection_result, detection_lock,
                             swap_input, swap_output, swap_lock,
                             enhancement_input, enhancement_output, enhancement_lock,
                             tick_rate_holder,
                             config: Optional[ProcessingConfig] = None):
    """Processing thread (consumer): takes raw frames from capture_queue,
    reads the latest detection result from the shared detection_result dict,
    applies face swap/enhancement, and puts results into processed_queue.

    Face detection is no longer performed here — it runs concurrently in
    _detection_thread_func and the most recent result is consumed lock-free
    (under a brief lock copy) so the swap loop never blocks on detection."""
    if config is None:
        config = build_config_from_globals()
    source_image = None
    last_source_path = None
    tick_count = 0
    tick_prev_time = time.time()
    tick_update_interval = 0.5
    prev_processed_frame = None
    rife_warned = False
    half_rate_warned = False
    frame_counter = 0
    enhancer_frame_counter = 0
    prev_enhanced_faces = None  # faces from the last submitted enhancement frame
    swap_seq = 0
    last_consumed_swap_seq = -1
    latest_swapped_frame = None
    enhancement_seq = 0
    last_consumed_enh_seq = -1
    latest_enhanced_frame = None

    while not stop_event.is_set():
        try:
            frame = capture_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        # Snapshot mutable globals for this frame — ensures consistent config
        # even if the UI thread toggles settings mid-frame.
        snap_live_mirror = modules.globals.live_mirror
        snap_half_rate = modules.globals.half_rate_processing
        snap_keyframe_interval = max(2, modules.globals.keyframe_interval)
        snap_map_faces = modules.globals.map_faces
        snap_source_path = modules.globals.source_path
        snap_many_faces = modules.globals.many_faces
        snap_motion_adaptive = modules.globals.motion_adaptive_enhancement
        snap_iou_thresh = modules.globals.motion_adaptive_iou_threshold
        snap_cos_thresh = modules.globals.motion_adaptive_cosine_threshold
        snap_enh_interval = max(1, modules.globals.enhancer_skip_interval)
        snap_rife_enabled = modules.globals.rife_enabled
        snap_rife_multiplier = modules.globals.rife_multiplier
        snap_virtual_cam = modules.globals.virtual_cam

        # Processors are fixed at session start (config snapshot).
        # Changing processors requires restarting the webcam session.
        frame_processors = get_frame_processors_modules(config.frame_processors)

        temp_frame = frame

        if snap_live_mirror:
            temp_frame = gpu_flip(temp_frame, 1)

        # Publish the mirrored frame for the detection thread to pick up
        with detection_lock:
            latest_frame_holder[0] = temp_frame

        # Half-rate processing: run face processing only on keyframes
        half_rate_enabled = snap_half_rate
        keyframe_interval = snap_keyframe_interval
        frame_counter += 1
        is_keyframe = (frame_counter % keyframe_interval) == 1
        skip_face_processing = half_rate_enabled and not is_keyframe

        if not skip_face_processing:
            if not snap_map_faces:
                current_source_path = snap_source_path
                if current_source_path and current_source_path != last_source_path:
                    last_source_path = current_source_path
                    source_image = get_one_face(cv2.imread(current_source_path))

                # Read latest detection results — brief lock copy so we don't
                # block the detection thread longer than necessary
                with detection_lock:
                    cached_target_face = detection_result.get('target_face')
                    cached_many_faces = detection_result.get('many_faces')

                # Build a face list from cached detection for enhancer reuse
                cached_faces_list = None
                if cached_many_faces:
                    cached_faces_list = cached_many_faces
                elif cached_target_face is not None:
                    cached_faces_list = [cached_target_face]

                # Enhancer skip-frame: motion-adaptive or fixed-interval
                motion_adaptive = snap_motion_adaptive
                iou_thresh = snap_iou_thresh
                cos_thresh = snap_cos_thresh
                if motion_adaptive and cached_faces_list is not None and prev_enhanced_faces is not None:
                    skip_enhancer = faces_are_similar(
                        cached_faces_list, prev_enhanced_faces, iou_thresh, cos_thresh
                    )
                else:
                    enhancer_frame_counter += 1
                    enh_interval = snap_enh_interval
                    skip_enhancer = enh_interval > 1 and (enhancer_frame_counter % enh_interval) != 1

                for frame_processor in frame_processors:
                    if frame_processor.NAME in _ENHANCER_NAMES:
                        if _is_enhancer_enabled(frame_processor):
                            if not skip_enhancer:
                                enhancement_seq += 1
                                prev_enhanced_faces = cached_faces_list
                                with enhancement_lock:
                                    enhancement_input[0] = {
                                        'frame': temp_frame.copy(),
                                        'faces': cached_faces_list,
                                        'map_faces': False,
                                        'processor': frame_processor,
                                        'seq': enhancement_seq,
                                    }
                            with enhancement_lock:
                                enh_out = enhancement_output[0]
                            if enh_out is not None and enh_out['seq'] != last_consumed_enh_seq:
                                last_consumed_enh_seq = enh_out['seq']
                                latest_enhanced_frame = enh_out['frame']
                            if latest_enhanced_frame is not None:
                                temp_frame = latest_enhanced_frame
                        else:
                            latest_enhanced_frame = None
                            prev_enhanced_faces = None
                            with enhancement_lock:
                                enhancement_input[0] = None
                                enhancement_output[0] = None
                    elif frame_processor.NAME == "DLC.FACE-SWAPPER":
                        swap_seq += 1
                        with swap_lock:
                            swap_input[0] = {
                                'frame': temp_frame.copy(),
                                'source_face': source_image,
                                'target_face': cached_target_face,
                                'many_faces': cached_many_faces if snap_many_faces else None,
                                'processor': frame_processor,
                                'map_faces': False,
                                'seq': swap_seq,
                                'config': config,
                            }
                        with swap_lock:
                            swap_out = swap_output[0]
                        if swap_out is not None and swap_out['seq'] != last_consumed_swap_seq:
                            last_consumed_swap_seq = swap_out['seq']
                            latest_swapped_frame = swap_out['frame']
                        if latest_swapped_frame is not None:
                            temp_frame = latest_swapped_frame
                    else:
                        temp_frame = frame_processor.process_frame(source_image, temp_frame)
            else:
                # In map_faces mode there is no fixed target file; clear globals for UI
                # consistency (legacy behaviour kept as side-effect).
                modules.globals.target_path = None

                # Enhancer skip-frame for map_faces path
                enhancer_frame_counter += 1
                enh_interval = snap_enh_interval
                skip_enhancer = enh_interval > 1 and (enhancer_frame_counter % enh_interval) != 1

                for frame_processor in frame_processors:
                    if frame_processor.NAME in _ENHANCER_NAMES:
                        if _is_enhancer_enabled(frame_processor):
                            if not skip_enhancer:
                                enhancement_seq += 1
                                with enhancement_lock:
                                    enhancement_input[0] = {
                                        'frame': temp_frame.copy(),
                                        'faces': None,
                                        'map_faces': True,
                                        'processor': frame_processor,
                                        'seq': enhancement_seq,
                                    }
                            with enhancement_lock:
                                enh_out = enhancement_output[0]
                            if enh_out is not None and enh_out['seq'] != last_consumed_enh_seq:
                                last_consumed_enh_seq = enh_out['seq']
                                latest_enhanced_frame = enh_out['frame']
                            if latest_enhanced_frame is not None:
                                temp_frame = latest_enhanced_frame
                        else:
                            latest_enhanced_frame = None
                            with enhancement_lock:
                                enhancement_input[0] = None
                                enhancement_output[0] = None
                    elif frame_processor.NAME == "DLC.FACE-SWAPPER":
                        swap_seq += 1
                        with swap_lock:
                            swap_input[0] = {
                                'frame': temp_frame.copy(),
                                'source_face': None,
                                'target_face': None,
                                'many_faces': None,
                                'processor': frame_processor,
                                'map_faces': True,
                                'seq': swap_seq,
                                'config': config,
                            }
                        with swap_lock:
                            swap_out = swap_output[0]
                        if swap_out is not None and swap_out['seq'] != last_consumed_swap_seq:
                            last_consumed_swap_seq = swap_out['seq']
                            latest_swapped_frame = swap_out['frame']
                        if latest_swapped_frame is not None:
                            temp_frame = latest_swapped_frame
                    else:
                        temp_frame = frame_processor.process_frame(None, temp_frame)
        else:
            # Skip frame: hold the last processed frame to avoid blending swapped/raw content.
            # Interpolating against a raw (unswapped) frame produces visible face-flicker artifacts.
            # When RIFE is enabled, the normal RIFE section below will generate smooth intermediates
            # between consecutive keyframes as they arrive (keyframe N → keyframe N+interval).
            if prev_processed_frame is not None:
                temp_frame = prev_processed_frame

        # RIFE frame interpolation: emit intermediate frames between consecutive keyframes.
        # In half-rate mode this fires on keyframes, bridging the gap since the previous keyframe
        # (which may be keyframe_interval camera frames back). Skip frames are held above and
        # never reach this block.
        rife_enabled = snap_rife_enabled
        if rife_enabled and prev_processed_frame is not None and not skip_face_processing:
            if not has_native_binding():
                if not rife_warned:
                    print("[DLC.RIFE] Native binding not available — live interpolation disabled")
                    rife_warned = True
            else:
                # In half-rate mode, generate exactly keyframe_interval-1 intermediates so
                # every skip slot between consecutive keyframes gets a smooth interpolated
                # frame. Outside half-rate mode, honour the user's rife_multiplier setting.
                if half_rate_enabled:
                    multiplier = keyframe_interval
                else:
                    multiplier = snap_rife_multiplier
                intermediates = interpolate_frame_pair(
                    prev_processed_frame, temp_frame, multiplier=multiplier,
                    should_stop=stop_event.is_set,
                )
                for interp_frame in intermediates:
                    tick_count += 1
                    if snap_virtual_cam:
                        virtual_cam.send(interp_frame)
                    try:
                        processed_queue.put_nowait(interp_frame)
                    except queue.Full:
                        try:
                            processed_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            processed_queue.put_nowait(interp_frame)
                        except queue.Full:
                            pass

        # Update prev_processed_frame on keyframes; keep unchanged on skip frames
        if not skip_face_processing:
            if rife_enabled or half_rate_enabled:
                prev_processed_frame = temp_frame.copy()
            else:
                prev_processed_frame = None
        # else (skip frame): prev_processed_frame stays set to the last keyframe output

        # Track processing tick rate (written to shared holder; read by display loop)
        tick_count += 1
        tick_now = time.time()
        if tick_now - tick_prev_time >= tick_update_interval:
            tick_rate_holder[0] = tick_count / (tick_now - tick_prev_time)
            tick_count = 0
            tick_prev_time = tick_now

        # When RIFE is active and this is a skip frame, suppress the duplicate emission.
        # RIFE will generate smooth intermediates on the next keyframe to fill this slot.
        # Emitting the dup would pollute the 4-slot queue and appear between interp frames,
        # creating a visible hold→jump pattern instead of smooth motion.
        if skip_face_processing and rife_enabled:
            continue

        # Send full-resolution processed frame to virtual camera if enabled
        if snap_virtual_cam:
            virtual_cam.send(temp_frame)

        # Put processed frame into output queue, dropping old frames if full
        try:
            processed_queue.put_nowait(temp_frame)
        except queue.Full:
            try:
                processed_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                processed_queue.put_nowait(temp_frame)
            except queue.Full:
                pass


def create_webcam_preview(camera_index: int, config: Optional[ProcessingConfig] = None):
    global _active_stop_event, _session_ready

    from modules import ui as _ui
    from modules.ui import (
        PREVIEW, ROOT,
        PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT,
        update_status, fit_image_to_size, toggle_preview,
    )
    from modules.core import destroy

    if config is None:
        config = build_config_from_globals()

    # If a session is already running, stop it and defer start until cleanup finishes.
    if _active_stop_event is not None and not _active_stop_event.is_set():
        _active_stop_event.set()

    if not _session_ready.is_set():
        def _retry():
            if _session_ready.is_set():
                create_webcam_preview(camera_index, config)
            else:
                ROOT.after(50, _retry)
        ROOT.after(50, _retry)
        return

    _session_ready.clear()

    set_det_size(FaceAnalyser.LIVE_DET_SIZE)

    cap = VideoCapturer(camera_index)
    if not cap.start(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT, 60):
        set_det_size(FaceAnalyser.DEFAULT_DET_SIZE)
        _session_ready.set()
        update_status("Failed to start camera")
        return

    if _ui._preview_embedded:
        pass  # embedded panel is always visible in the right column
    else:
        _ui.preview_label.configure(width=PREVIEW_DEFAULT_WIDTH, height=PREVIEW_DEFAULT_HEIGHT)
        # During live mode the preview window IS the app — X should quit entirely.
        PREVIEW.protocol("WM_DELETE_WINDOW", destroy)
        PREVIEW.deiconify()

    # Start virtual camera if enabled
    if config.virtual_cam:
        if not virtual_cam.start(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT):
            update_status("Virtual camera failed to start — check logs")

    # Queues for decoupling capture from processing and processing from display.
    # Small maxsize ensures we always work on recent frames and drop stale ones.
    capture_queue = queue.Queue(maxsize=2)
    processed_queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()
    _active_stop_event = stop_event

    # Shared state for display overlay: processing tick rate written by the
    # processing thread, display FPS tracked in the display loop.
    tick_rate_holder = [0.0]   # frames/sec produced by processing thread
    latest_display_frame = [None]  # last frame shown; held when queue is empty
    display_fps_state = [0, time.time(), 0.0]  # [frame_count, prev_time, fps]

    # Shared state for the producer-consumer detection pipeline.
    # latest_frame_holder[0] is the most recent raw frame for the detection
    # thread to consume; detection_result holds the last detected faces for
    # the processing thread to read.  Both are guarded by detection_lock.
    detection_lock = threading.Lock()
    latest_frame_holder = [None]  # one-element list so inner functions can rebind
    detection_result = {'target_face': None, 'many_faces': None}

    # EMA smoother for landmark/bbox jitter reduction (live mode only).
    # Always created so the detection thread can accept the parameter;
    # smoothing is only applied when modules.globals.landmark_smoothing is True.
    landmark_smoother = LandmarkSmoother(alpha=modules.globals.landmark_smoothing_alpha)

    # Shared state for the async swap pipeline.
    swap_lock = threading.Lock()
    swap_input = [None]          # single-slot holder for swap requests
    swap_output = [None]         # single-slot holder for swap results

    # Shared state for the async enhancement pipeline.
    enhancement_lock = threading.Lock()
    enhancement_input = [None]   # single-slot holder for enhancement requests
    enhancement_output = [None]  # single-slot holder for enhancement results

    # Start capture thread
    cap_thread = threading.Thread(
        target=_capture_thread_func,
        args=(cap, capture_queue, stop_event),
        daemon=True,
    )
    cap_thread.start()

    # Start detection thread — runs face detection asynchronously on the
    # latest raw frame so the processing/swap thread never blocks on it.
    det_thread = threading.Thread(
        target=_detection_thread_func,
        args=(latest_frame_holder, detection_result, detection_lock, stop_event, config,
              landmark_smoother),
        daemon=True,
    )
    det_thread.start()

    # Start swap thread — runs face swap ONNX inference asynchronously so the
    # processing thread never blocks on swap computation.
    swap_thread = threading.Thread(
        target=_swap_thread_func,
        args=(swap_input, swap_output, swap_lock, stop_event),
        daemon=True,
    )
    swap_thread.start()

    # Start enhancement thread — runs face enhancement asynchronously so the
    # processing thread never blocks on expensive GFPGAN/GPEN inference.
    enh_thread = threading.Thread(
        target=_enhancement_thread_func,
        args=(enhancement_input, enhancement_output, enhancement_lock, stop_event),
        daemon=True,
    )
    enh_thread.start()

    # Start processing thread
    proc_thread = threading.Thread(
        target=_processing_thread_func,
        args=(capture_queue, processed_queue, stop_event,
              latest_frame_holder, detection_result, detection_lock,
              swap_input, swap_output, swap_lock,
              enhancement_input, enhancement_output, enhancement_lock,
              tick_rate_holder, config),
        daemon=True,
    )
    proc_thread.start()

    def _cleanup():
        # Signal threads and hide/restore the window immediately so the UI stays
        # responsive.  Blocking operations (join/release/det-size reset) run
        # on a daemon thread so the main thread is never stalled.
        stop_event.set()
        if not _ui._preview_embedded:
            PREVIEW.protocol("WM_DELETE_WINDOW", toggle_preview)
            PREVIEW.withdraw()

        def _background_cleanup():
            cleanup_rife()
            cap_thread.join(timeout=2.0)
            det_thread.join(timeout=2.0)
            swap_thread.join(timeout=2.0)
            enh_thread.join(timeout=2.0)
            proc_thread.join(timeout=2.0)
            cap.release()
            virtual_cam.stop()
            set_det_size(FaceAnalyser.DEFAULT_DET_SIZE)
            _session_ready.set()

        threading.Thread(target=_background_cleanup, daemon=True).start()

    def _display_next_frame():
        """Non-blocking display step — reschedules itself via ROOT.after().

        Tracks actual displayed FPS independently of the processing tick rate.
        Holds the last known frame when the queue is empty so the display rate
        is decoupled from the camera input rate.
        """
        if stop_event.is_set():
            _cleanup()
            return
        # In pop-out mode, stop when the PREVIEW window is withdrawn
        if not _ui._preview_embedded and PREVIEW.state() == "withdrawn":
            _cleanup()
            return

        try:
            frame = processed_queue.get_nowait()
            latest_display_frame[0] = frame
            # Only count unique new frames toward display FPS
            display_fps_state[0] += 1
            now = time.time()
            if now - display_fps_state[1] >= 0.5:
                display_fps_state[2] = display_fps_state[0] / (now - display_fps_state[1])
                display_fps_state[0] = 0
                display_fps_state[1] = now
        except queue.Empty:
            frame = latest_display_frame[0]

        if frame is None:
            ROOT.after(max(1, 1000 // modules.globals.live_max_fps), _display_next_frame)
            return

        if _ui._preview_embedded and _ui._embedded_label:
            # Embedded panel: always scale to fit so the window never grows
            w = _ui._embedded_label.winfo_width()
            h = _ui._embedded_label.winfo_height()
            frame = fit_image_to_size(frame, w, h)
        elif modules.globals.live_resizable:
            # Pop-out window with resizable enabled: scale to window size
            w = PREVIEW.winfo_width()
            h = PREVIEW.winfo_height()
            frame = fit_image_to_size(frame, w, h)
        # Pop-out + live_resizable=False: display at native camera resolution

        image = Image.fromarray(gpu_cvt_color(frame, cv2.COLOR_BGR2RGB))

        if modules.globals.show_fps:
            draw = ImageDraw.Draw(image)
            overlay = f"FPS: {display_fps_state[2]:.1f}  Tick: {tick_rate_holder[0]:.1f}"
            # Shadow for readability on any background
            draw.text((11, 11), overlay, fill=(0, 0, 0))
            draw.text((10, 10), overlay, fill=(0, 255, 0))

        # Dynamic label — picks up latest assignment even if user pops out/in during live
        lbl = _ui.preview_label
        lbl._ctk_img = ctk.CTkImage(image, size=image.size)
        lbl.configure(image=lbl._ctk_img)

        ROOT.after(max(1, 1000 // modules.globals.live_max_fps), _display_next_frame)

    # Kick off the non-blocking display loop
    ROOT.after(max(1, 1000 // modules.globals.live_max_fps), _display_next_frame)


def webcam_preview(root: ctk.CTk, camera_index: int, config: Optional[ProcessingConfig] = None):
    from modules.ui import POPUP_LIVE, update_status
    from modules.mapping_list import MAPPING_LIST
    from modules.face_map_store import STORE as _MAP_STORE

    if config is None:
        config = build_config_from_globals()

    if POPUP_LIVE is not None and POPUP_LIVE.winfo_exists():
        update_status("Source x Target Mapper is already open.")
        POPUP_LIVE.focus()
        return

    effective_map = MAPPING_LIST.effective_map_faces()

    if not effective_map:
        # Single source, no pins — simple face swap mode
        if MAPPING_LIST.source_count() == 0:
            update_status("Please select a source image first")
            return
        # Sync mapping list to store for consistency
        MAPPING_LIST.sync_to_store(_MAP_STORE)
        create_webcam_preview(camera_index, config=config)
    else:
        # Multiple sources or pins — map faces mode
        MAPPING_LIST.sync_to_store(_MAP_STORE)
        if _MAP_STORE.has_valid_map():
            _MAP_STORE.simplify()
        create_webcam_preview(camera_index, config=config)
