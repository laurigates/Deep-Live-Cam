import cv2
import time
import queue
import threading
from PIL import Image
import customtkinter as ctk

import modules.globals
from modules import virtual_cam
from modules.gpu_processing import gpu_cvt_color, gpu_flip
from modules.face_analyser import get_one_face, get_many_faces, set_det_size, _LIVE_DET_SIZE, _DEFAULT_DET_SIZE, faces_are_similar
from modules.processors.frame.core import get_frame_processors_modules
from modules.rife_interpolation import has_native_binding, interpolate_frame_pair
from modules.video_capture import VideoCapturer


# DETECT_EVERY_N is kept for backward-compatibility with any external imports
# but is no longer used by the processing thread — detection now runs in its
# own dedicated thread.
DETECT_EVERY_N = 2

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


def _is_enhancer_enabled(processor) -> bool:
    """Check if an enhancer processor is toggled on in the UI."""
    key = _ENHANCER_UI_KEYS.get(processor.NAME)
    return key is not None and modules.globals.fp_ui.get(key, False)


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


def _detection_thread_func(latest_frame_holder, detection_result, detection_lock, stop_event):
    """Detection thread (producer): continuously reads the most recently
    captured raw frame and runs face detection on it, storing results in
    *detection_result* under *detection_lock*.

    latest_frame_holder is a one-element list [frame | None] written by the
    processing thread so the detection thread always works on the newest frame
    without queuing overhead.  The detection thread never touches Tkinter
    widgets — all UI updates go through ROOT.after() in the display loop.
    """
    while not stop_event.is_set():
        with detection_lock:
            frame = latest_frame_holder[0]

        if frame is None:
            time.sleep(0.005)
            continue

        if modules.globals.many_faces:
            many = get_many_faces(frame)
            with detection_lock:
                detection_result['target_face'] = None
                detection_result['many_faces'] = many
        else:
            face = get_one_face(frame)
            with detection_lock:
                detection_result['target_face'] = face
                detection_result['many_faces'] = None


def _swap_thread_func(swap_input, swap_output, swap_lock, stop_event):
    """Swap thread: runs face swap ONNX inference asynchronously.

    Follows the same single-slot holder pattern as detection and enhancement.
    Reads from swap_input[0], writes to swap_output[0].
    Input holder dict: {'frame', 'source_face', 'target_face', 'many_faces',
                        'processor', 'map_faces', 'seq'}
    Output holder dict: {'frame', 'seq'}
    """
    last_processed_seq = -1

    while not stop_event.is_set():
        with swap_lock:
            inp = swap_input[0]

        if inp is None:
            time.sleep(0.005)
            continue

        seq = inp['seq']
        if seq == last_processed_seq:
            time.sleep(0.005)
            continue

        frame = inp['frame']
        processor = inp['processor']

        if inp['map_faces']:
            frame = processor.process_frame(None, frame)
        else:
            source_face = inp['source_face']
            many_faces_list = inp['many_faces']
            target_face = inp['target_face']

            swapped_bboxes = []
            if many_faces_list:
                opacity = getattr(modules.globals, "opacity", 1.0)
                result = frame if opacity >= 1.0 else frame.copy()
                for t_face in many_faces_list:
                    result = processor.swap_face(source_face, t_face, result)
                    if hasattr(t_face, 'bbox') and t_face.bbox is not None:
                        swapped_bboxes.append(t_face.bbox.astype(int))
                frame = result
            elif target_face is not None:
                frame = processor.swap_face(source_face, target_face, frame)
                if hasattr(target_face, 'bbox') and target_face.bbox is not None:
                    swapped_bboxes.append(target_face.bbox.astype(int))

            frame = processor.apply_post_processing(frame, swapped_bboxes)

        last_processed_seq = seq

        with swap_lock:
            swap_output[0] = {'frame': frame, 'seq': seq}


def _enhancement_thread_func(enhancement_input, enhancement_output,
                              enhancement_lock, stop_event):
    """Enhancement thread: runs face enhancement (GFPGAN/GPEN) asynchronously.

    Follows the same single-slot holder pattern as the detection thread.
    Reads from enhancement_input[0], writes to enhancement_output[0].
    Input holder dict: {'frame', 'faces', 'map_faces', 'processor', 'seq'}
    Output holder dict: {'frame', 'seq'}
    """
    last_processed_seq = -1

    while not stop_event.is_set():
        with enhancement_lock:
            inp = enhancement_input[0]

        if inp is None:
            time.sleep(0.005)
            continue

        seq = inp['seq']
        if seq == last_processed_seq:
            time.sleep(0.005)
            continue

        processor = inp['processor']
        frame = inp['frame']
        faces = inp['faces']
        map_faces = inp['map_faces']

        if map_faces:
            enhanced = processor.process_frame_v2(frame)
        else:
            enhanced = processor.process_frame(None, frame, faces=faces)

        last_processed_seq = seq

        with enhancement_lock:
            enhancement_output[0] = {'frame': enhanced, 'seq': seq}


def _processing_thread_func(capture_queue, processed_queue, stop_event,
                             latest_frame_holder, detection_result, detection_lock,
                             swap_input, swap_output, swap_lock,
                             enhancement_input, enhancement_output, enhancement_lock):
    """Processing thread (consumer): takes raw frames from capture_queue,
    reads the latest detection result from the shared detection_result dict,
    applies face swap/enhancement, and puts results into processed_queue.

    Face detection is no longer performed here — it runs concurrently in
    _detection_thread_func and the most recent result is consumed lock-free
    (under a brief lock copy) so the swap loop never blocks on detection."""
    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
    source_image = None
    last_source_path = None
    prev_time = time.time()
    fps_update_interval = 0.5
    frame_count = 0
    fps = 0
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

        temp_frame = frame

        if modules.globals.live_mirror:
            temp_frame = gpu_flip(temp_frame, 1)

        # Publish the mirrored frame for the detection thread to pick up
        with detection_lock:
            latest_frame_holder[0] = temp_frame

        # Half-rate processing: run face processing only on keyframes
        half_rate_enabled = getattr(modules.globals, "half_rate_processing", False)
        keyframe_interval = max(2, getattr(modules.globals, "keyframe_interval", 2))
        frame_counter += 1
        is_keyframe = (frame_counter % keyframe_interval) == 1
        skip_face_processing = half_rate_enabled and not is_keyframe

        if not skip_face_processing:
            if not modules.globals.map_faces:
                if modules.globals.source_path and modules.globals.source_path != last_source_path:
                    last_source_path = modules.globals.source_path
                    source_image = get_one_face(cv2.imread(modules.globals.source_path))

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
                motion_adaptive = getattr(modules.globals, 'motion_adaptive_enhancement', False)
                iou_thresh = getattr(modules.globals, 'motion_adaptive_iou_threshold', 0.9)
                cos_thresh = getattr(modules.globals, 'motion_adaptive_cosine_threshold', 0.95)
                if motion_adaptive and cached_faces_list is not None:
                    skip_enhancer = faces_are_similar(
                        cached_faces_list, prev_enhanced_faces, iou_thresh, cos_thresh
                    )
                else:
                    enhancer_frame_counter += 1
                    enh_interval = max(1, getattr(modules.globals, "enhancer_skip_interval", 1))
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
                                'many_faces': cached_many_faces if modules.globals.many_faces else None,
                                'processor': frame_processor,
                                'map_faces': False,
                                'seq': swap_seq,
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
                modules.globals.target_path = None

                # Enhancer skip-frame for map_faces path
                enhancer_frame_counter += 1
                enh_interval = max(1, getattr(modules.globals, "enhancer_skip_interval", 1))
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
        rife_enabled = getattr(modules.globals, "rife_enabled", False)
        if rife_enabled and prev_processed_frame is not None and not skip_face_processing:
            if not rife_warned and not has_native_binding():
                print("[DLC.RIFE] Native binding not available — live interpolation disabled")
                rife_warned = True
            else:
                multiplier = getattr(modules.globals, "rife_multiplier", 2)
                intermediates = interpolate_frame_pair(
                    prev_processed_frame, temp_frame, multiplier=multiplier
                )
                for interp_frame in intermediates:
                    frame_count += 1
                    if modules.globals.virtual_cam:
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

        # Calculate and display FPS
        current_time = time.time()
        frame_count += 1
        if current_time - prev_time >= fps_update_interval:
            fps = frame_count / (current_time - prev_time)
            frame_count = 0
            prev_time = current_time

        if modules.globals.show_fps:
            cv2.putText(
                temp_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        # Send full-resolution processed frame to virtual camera if enabled
        if modules.globals.virtual_cam:
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


def create_webcam_preview(camera_index: int):
    from modules.ui import (
        preview_label, PREVIEW, ROOT,
        PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT,
        update_status, fit_image_to_size,
    )

    set_det_size(_LIVE_DET_SIZE)

    cap = VideoCapturer(camera_index)
    if not cap.start(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT, 60):
        set_det_size(_DEFAULT_DET_SIZE)
        update_status("Failed to start camera")
        return

    preview_label.configure(width=PREVIEW_DEFAULT_WIDTH, height=PREVIEW_DEFAULT_HEIGHT)
    PREVIEW.deiconify()

    # Start virtual camera if enabled
    if modules.globals.virtual_cam:
        if not virtual_cam.start(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT):
            update_status("Virtual camera failed to start — check logs")

    # Queues for decoupling capture from processing and processing from display.
    # Small maxsize ensures we always work on recent frames and drop stale ones.
    capture_queue = queue.Queue(maxsize=2)
    processed_queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()

    # Shared state for the producer-consumer detection pipeline.
    # latest_frame_holder[0] is the most recent raw frame for the detection
    # thread to consume; detection_result holds the last detected faces for
    # the processing thread to read.  Both are guarded by detection_lock.
    detection_lock = threading.Lock()
    latest_frame_holder = [None]  # one-element list so inner functions can rebind
    detection_result = {'target_face': None, 'many_faces': None}

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
        args=(latest_frame_holder, detection_result, detection_lock, stop_event),
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
              enhancement_input, enhancement_output, enhancement_lock),
        daemon=True,
    )
    proc_thread.start()

    def _cleanup():
        stop_event.set()
        cap_thread.join(timeout=2.0)
        det_thread.join(timeout=2.0)
        swap_thread.join(timeout=2.0)
        enh_thread.join(timeout=2.0)
        proc_thread.join(timeout=2.0)
        cap.release()
        virtual_cam.stop()
        set_det_size(_DEFAULT_DET_SIZE)
        PREVIEW.withdraw()

    def _display_next_frame():
        """Non-blocking display step — reschedules itself via ROOT.after()."""
        if stop_event.is_set() or PREVIEW.state() == "withdrawn":
            _cleanup()
            return


        try:
            temp_frame = processed_queue.get_nowait()
        except queue.Empty:
            ROOT.after(16, _display_next_frame)
            return

        if modules.globals.live_resizable:
            temp_frame = fit_image_to_size(
                temp_frame, PREVIEW.winfo_width(), PREVIEW.winfo_height()
            )
        # live_resizable=False: display at native camera resolution

        image = gpu_cvt_color(temp_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)

        ROOT.after(16, _display_next_frame)

    # Kick off the non-blocking display loop
    ROOT.after(16, _display_next_frame)


def webcam_preview(root: ctk.CTk, camera_index: int):
    from modules.ui import POPUP_LIVE, update_status
    from modules.ui_mapper import create_source_target_popup_for_webcam

    if POPUP_LIVE is not None and POPUP_LIVE.winfo_exists():
        update_status("Source x Target Mapper is already open.")
        POPUP_LIVE.focus()
        return

    if not modules.globals.map_faces:
        if modules.globals.source_path is None:
            update_status("Please select a source image first")
            return
        create_webcam_preview(camera_index)
    else:
        with modules.globals.MAP_LOCK:
            modules.globals.source_target_map = []
        create_source_target_popup_for_webcam(
            root, modules.globals.source_target_map, camera_index
        )
