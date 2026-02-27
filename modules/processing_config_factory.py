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
        # Face Swapper Options
        face_swapper_enabled=modules.globals.face_swapper_enabled,
        opacity=modules.globals.opacity,
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
        webcam_preview_running=modules.globals.webcam_preview_running,
        live_mirror=modules.globals.live_mirror,
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
    """
    Build ProcessingConfig directly from CLI arguments.

    This function will replace build_config_from_globals() once all modules
    are updated to accept injected configuration.

    Args:
        args: Parsed command-line arguments (from argparse)

    Returns:
        ProcessingConfig instance configured from CLI args
    """
    # This is a placeholder for the future CLI-based config builder.
    # For now, it would extract fields from args and return a ProcessingConfig.
    # Example (once implemented):
    #   return ProcessingConfig(
    #       execution_providers=args.execution_provider or ['cpu'],
    #       source_path=args.source,
    #       target_path=args.target,
    #       ... etc
    #   )
    raise NotImplementedError("CLI-based config builder not yet implemented")
