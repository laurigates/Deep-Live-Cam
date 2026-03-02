"""Factory functions to build ProcessingConfig from CLI args, UI state, or globals.

This module handles the transition from global variables to injected configuration.
Initially, it creates ProcessingConfig from modules.globals for compatibility.
Eventually, it will build config directly from CLI args/UI state.
"""
from typing import Optional
from modules.processing_config import ProcessingConfig
import modules.globals


def build_config_from_globals() -> ProcessingConfig:
    """
    Build ProcessingConfig from current module-level globals.

    This is a compatibility function for the migration phase. As modules are
    updated to accept ProcessingConfig, this function bridges the gap by
    extracting all global variables into a single config object.

    Eventually, this function will be replaced with factory functions that
    build config directly from CLI args or UI state.
    """
    return ProcessingConfig(
        # Execution Configuration
        execution_providers=modules.globals.execution_providers.copy(),
        execution_threads=modules.globals.execution_threads,
        max_memory=modules.globals.max_memory,
        headless=modules.globals.headless,
        log_level=modules.globals.log_level,
        coreml_compute_units=modules.globals.coreml_compute_units,
        # Paths
        source_path=modules.globals.source_path,
        target_path=modules.globals.target_path,
        output_path=modules.globals.output_path,
        # Frame Processing
        frame_processors=modules.globals.frame_processors.copy(),
        # Face Detection
        face_confidence_threshold=modules.globals.FACE_CONFIDENCE_THRESHOLD,
        detection_interval=modules.globals.DETECTION_INTERVAL,
        detection_cache_size=modules.globals.DETECTION_CACHE_SIZE,
        # Processing Options
        many_faces=modules.globals.many_faces,
        map_faces=modules.globals.map_faces,
        keep_fps=modules.globals.keep_fps,
        keep_audio=modules.globals.keep_audio,
        keep_frames=modules.globals.keep_frames,
        use_png_frames=modules.globals.use_png_frames,
        # Face Swapper Options
        face_swapper_enabled=modules.globals.face_swapper_enabled,
        face_swap_model=getattr(modules.globals, 'face_swap_model', 'inswapper'),
        opacity=max(0.0, min(1.0, getattr(modules.globals, 'opacity', 1.0))),
        sharpness=modules.globals.sharpness,
        prepaste_upscale=modules.globals.prepaste_upscale,
        color_correction=modules.globals.color_correction,
        poisson_blend=modules.globals.poisson_blend,
        # Face Enhancer
        codeformer_fidelity=modules.globals.codeformer_fidelity,
        # Mouth Mask
        mouth_mask=modules.globals.mouth_mask,
        show_mouth_mask_box=modules.globals.show_mouth_mask_box,
        mouth_feather_radius=modules.globals.MOUTH_FEATHER_RADIUS,
        mask_feather_ratio=modules.globals.mask_feather_ratio,
        mask_down_size=modules.globals.mask_down_size,
        mask_size=modules.globals.mask_size,
        mouth_mask_size=modules.globals.mouth_mask_size,
        eyes_mask_size=modules.globals.eyes_mask_size,
        eyebrows_mask_size=modules.globals.eyebrows_mask_size,
        # Frame Interpolation
        enable_interpolation=modules.globals.enable_interpolation,
        interpolation_weight=modules.globals.interpolation_weight,
        rife_enabled=modules.globals.rife_enabled,
        rife_model=modules.globals.rife_model,
        rife_multiplier=modules.globals.rife_multiplier,
        # Half-Rate Processing
        half_rate_processing=modules.globals.half_rate_processing,
        keyframe_interval=modules.globals.keyframe_interval,
        # Enhancer Skip-Frame
        enhancer_skip_interval=modules.globals.enhancer_skip_interval,
        # Live Mode
        live_enhance_size=modules.globals.live_enhance_size,
        motion_adaptive_enhancement=modules.globals.motion_adaptive_enhancement,
        motion_adaptive_iou_threshold=modules.globals.motion_adaptive_iou_threshold,
        motion_adaptive_cosine_threshold=modules.globals.motion_adaptive_cosine_threshold,
        landmark_smoothing=modules.globals.landmark_smoothing,
        landmark_smoothing_alpha=modules.globals.landmark_smoothing_alpha,
        webcam_preview_running=modules.globals.webcam_preview_running,
        live_mirror=modules.globals.live_mirror,
        live_resizable=modules.globals.live_resizable,
        live_max_fps=modules.globals.live_max_fps,
        show_fps=modules.globals.show_fps,
        virtual_cam=modules.globals.virtual_cam,
        # Video Output
        video_encoder=modules.globals.video_encoder,
        video_quality=modules.globals.video_quality,
        # NSFW Filter
        nsfw_filter=modules.globals.nsfw_filter,
        # UI State
        fp_ui=modules.globals.fp_ui.copy(),
        # map_lock is self-provisioned — FaceMapStore now owns face-map locking
    )


def build_config_from_cli_args(args) -> ProcessingConfig:
    """Build ProcessingConfig directly from CLI arguments.

    Args:
        args: Parsed command-line arguments (from argparse).  Deprecated fields
              (source_path_deprecated, cpu_cores_deprecated, gpu_vendor_deprecated,
              gpu_threads_deprecated) are honoured when present.

    Returns:
        ProcessingConfig instance configured from CLI args.
    """
    # Lazy import to avoid circular dependency (core imports this factory).
    from modules.core import decode_execution_providers, normalize_output_path

    source_path = args.source_path
    target_path = args.target_path
    output_path = normalize_output_path(source_path, target_path, args.output_path)
    execution_providers = decode_execution_providers(args.execution_provider or ['cpu'])
    execution_threads = args.execution_threads

    # Handle deprecated arguments
    if getattr(args, 'source_path_deprecated', None):
        source_path = args.source_path_deprecated
        output_path = normalize_output_path(source_path, target_path, args.output_path)
    if getattr(args, 'cpu_cores_deprecated', None):
        execution_threads = args.cpu_cores_deprecated
    if getattr(args, 'gpu_vendor_deprecated', None):
        if args.gpu_vendor_deprecated == 'apple':
            execution_providers = decode_execution_providers(['coreml'])
        elif args.gpu_vendor_deprecated == 'nvidia':
            execution_providers = decode_execution_providers(['cuda'])
        elif args.gpu_vendor_deprecated == 'amd':
            execution_providers = decode_execution_providers(['rocm'])
    if getattr(args, 'gpu_threads_deprecated', None):
        execution_threads = args.gpu_threads_deprecated

    frame_processors = list(args.frame_processor)
    fp_ui = {
        'face_enhancer': 'face_enhancer' in frame_processors,
        'face_enhancer_gpen256': 'face_enhancer_gpen256' in frame_processors,
        'face_enhancer_gpen512': 'face_enhancer_gpen512' in frame_processors,
        'face_enhancer_codeformer': 'face_enhancer_codeformer' in frame_processors,
    }

    return ProcessingConfig(
        source_path=source_path,
        target_path=target_path,
        output_path=output_path,
        frame_processors=frame_processors,
        use_png_frames=args.use_png_frames,
        keep_fps=args.keep_fps,
        keep_audio=args.keep_audio,
        keep_frames=args.keep_frames,
        many_faces=args.many_faces,
        mouth_mask=args.mouth_mask,
        nsfw_filter=args.nsfw_filter,
        map_faces=args.map_faces,
        video_encoder=args.video_encoder,
        video_quality=args.video_quality,
        live_mirror=args.live_mirror,
        live_resizable=args.live_resizable,
        virtual_cam=args.virtual_cam,
        max_memory=args.max_memory,
        coreml_compute_units=getattr(args, 'coreml_compute_units', 'ALL'),
        execution_providers=execution_providers,
        execution_threads=execution_threads,
        rife_enabled=args.rife_enabled,
        rife_model=args.rife_model,
        rife_multiplier=args.rife_multiplier,
        half_rate_processing=args.half_rate_processing,
        keyframe_interval=args.keyframe_interval,
        live_enhance_size=args.live_enhance_size,
        face_swap_model=getattr(args, 'face_swap_model', 'inswapper'),
        fp_ui=fp_ui,
        headless=bool(source_path or target_path or args.output_path),
    )
