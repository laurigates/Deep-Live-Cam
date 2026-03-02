"""ProcessingConfig — injectable configuration to replace module-level globals.

This module provides a dataclass that encapsulates all processing configuration,
enabling dependency injection instead of relying on module-level global variables.

Gradually transitioning from:
    modules.globals.many_faces
    modules.globals.execution_providers
    ...

To:
    config.many_faces
    config.execution_providers
    ...

This enables:
- Testing modules in isolation with different configs
- Supporting multiple simultaneous processing pipelines
- Thread-safe configuration without mutable shared state
- Clearer module dependencies (config is explicit parameter, not hidden global read)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import threading


@dataclass
class ProcessingConfig:
    """Complete processing configuration for face swapping and enhancement.

    This dataclass replaces the ~80 module-level variables in modules.globals.py.
    Create an instance from CLI args or UI state, then inject into processing modules.
    """

    # ======================== Execution Configuration ========================
    execution_providers: List[str] = field(default_factory=list)
    """ONNX Runtime providers, e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']"""

    execution_threads: Optional[int] = None
    """Number of threads for parallel frame processing"""

    max_memory: Optional[int] = None
    """Memory limit in GB (optional)"""

    headless: bool = False
    """Run without UI"""

    log_level: str = "error"
    """Logging level: 'debug', 'info', 'warning', 'error'"""

    coreml_compute_units: str = "ALL"
    """CoreML compute units for Apple Silicon: 'ALL' (ANE+GPU+CPU), 'CPUAndGPU', 'CPUOnly'"""

    # ======================== Input/Output Paths ========================
    source_path: Optional[str] = None
    """Path to source image or video for face swapping"""

    target_path: Optional[str] = None
    """Path to target image or video for face detection"""

    output_path: Optional[str] = None
    """Path where the processed output will be saved"""

    # ======================== Frame Processor Pipeline ========================
    frame_processors: List[str] = field(default_factory=list)
    """List of enabled frame processors: ['face_swapper', 'face_enhancer', ...]"""

    # ======================== Face Detection ========================
    face_confidence_threshold: float = 0.5
    """Minimum confidence score to accept a detected face"""

    detection_interval: float = 0.033
    """Face detection cache TTL in seconds (~30 FPS for live mode)"""

    detection_cache_size: int = 128
    """Maximum entries in face detection LRU cache"""

    face_analyser_det_size: tuple = (320, 320)
    """Detection model input size: (width, height)"""

    # ======================== Processing Options ========================
    many_faces: bool = False
    """Process all detected faces with default source face"""

    map_faces: bool = False
    """Use explicit source→target mappings instead of default"""

    keep_fps: bool = True
    """Preserve original video frame rate in output"""

    keep_audio: bool = True
    """Preserve original audio in output video"""

    keep_frames: bool = False
    """Keep temporary frame files after processing"""

    use_png_frames: bool = False
    """Use lossless PNG for intermediate frames (vs JPEG)"""

    # ======================== Face Swapper Options ========================
    face_swapper_enabled: bool = True
    """General toggle for the face swapper processor"""

    face_swap_model: str = 'inswapper'
    """Face swap model variant: 'inswapper' | 'ghost_256_v1' | 'ghost_256_v2' | 'ghost_256_v3' | 'hyperswap_256_1a' | 'hyperswap_256_1b' | 'hyperswap_256_1c'"""

    opacity: float = 1.0
    """Blend factor for swapped face (0.0 = original, 1.0 = fully swapped)"""

    sharpness: float = 0.0
    """Sharpness enhancement for swapped face (0.0-1.0+)"""

    prepaste_upscale: bool = True
    """Upscale swap crop before paste-back to reduce stretch artifact"""

    color_correction: bool = False
    """Enable color correction for swapped face"""

    color_correction_mode: str = 'none'
    """Color correction mode for swapped face crop: 'none', 'lab', or 'histogram'.
    'lab' applies LAB mean/std transfer; 'histogram' applies per-channel CDF matching."""

    poisson_blend: bool = False
    """Enable Poisson blending for smoother face swaps"""

    swap_color_transfer: bool = False
    """Apply LAB color transfer to swapped crop before paste-back"""

    occlusion_mask: bool = False
    """Preserve occluding objects (hands, glasses, microphones) during face swap using XSeg model"""

    # ======================== Paste-back Tuning ========================
    paste_diff_threshold: float = 10.0
    """Threshold for diff mask binarisation in swap paste-back"""

    paste_mask_threshold: float = 20.0
    """Threshold for white mask binarisation in swap paste-back"""

    paste_mask_erode_ratio: int = 10
    """Divisor for erosion kernel size (mask_size // ratio, min 10)"""

    paste_mask_blur_ratio: int = 20
    """Divisor for blur kernel size (mask_size // ratio, min 5)"""

    enhance_feather_fraction: float = 0.05
    """Border fraction for enhancer feathered paste-back mask"""

    # ======================== Face Enhancement Options ========================
    # CodeFormer fidelity (0.0 = max quality, 1.0 = max fidelity to source)
    codeformer_fidelity: float = 0.7

    # ======================== Mouth Mask Options ========================
    mouth_mask: bool = False
    """Enable mouth area masking/pasting"""

    show_mouth_mask_box: bool = False
    """Visualize mouth mask area (for debugging)"""

    mouth_feather_radius: int = 10
    """Pixel radius for mouth mask edge feathering"""

    mask_feather_ratio: int = 12
    """Denominator for feathering calculation (higher = smaller feather)"""

    mask_down_size: float = 0.1
    """Expansion factor for lower lip mask"""

    mask_size: float = 1.0
    """Expansion factor for upper lip mask"""

    mouth_mask_size: float = 1.0
    """Scale factor for mouth mask region"""

    eyes_mask_size: float = 1.0
    """Scale factor for eyes mask region"""

    eyebrows_mask_size: float = 1.0
    """Scale factor for eyebrows mask region"""

    # ======================== Frame Interpolation (RIFE) ========================
    enable_interpolation: bool = True
    """Toggle temporal smoothing"""

    interpolation_weight: float = 0.0
    """Blend weight for current frame (0.0-1.0). Lower = smoother."""

    rife_enabled: bool = False
    """Toggle RIFE frame interpolation for video output"""

    rife_model: str = "rife-v4.25-lite"
    """RIFE model: 'rife-v4.25' or 'rife-v4.25-lite'"""

    rife_multiplier: int = 2
    """Frame rate multiplier: 2 = double fps, 4 = quadruple fps"""

    # ======================== Half-Rate Processing ========================
    half_rate_processing: bool = False
    """Process every Nth frame; use RIFE to fill skipped frames"""

    keyframe_interval: int = 2
    """Run face processing every Nth frame (2-10)"""

    # ======================== Enhancer Skip-Frame ========================
    enhancer_skip_interval: int = 1
    """Enhance every Nth frame (1 = every, 2 = every other, etc.)"""

    # ======================== Live Mode Options ========================
    live_enhance_size: int = 256
    """Face alignment/paste resolution in live mode (smaller = faster)"""

    motion_adaptive_enhancement: bool = False
    """Skip enhancement when face hasn't moved significantly"""

    motion_adaptive_iou_threshold: float = 0.9
    """Min bbox IoU to reuse previous enhanced result"""

    motion_adaptive_cosine_threshold: float = 0.95
    """Min embedding cosine to reuse previous enhanced result"""

    landmark_smoothing: bool = False
    """Apply EMA smoothing to face bounding boxes and keypoints in live mode"""

    landmark_smoothing_alpha: float = 0.7
    """EMA weight for the current frame (0.0=full history, 1.0=no smoothing)"""

    webcam_preview_running: bool = False
    """Indicates if live webcam preview is active"""

    live_mirror: bool = False
    """Mirror the live preview horizontally"""

    live_resizable: bool = True
    """Allow the live preview window to be resized"""

    live_max_fps: int = 30
    """Maximum preview frame rate for live mode"""

    show_fps: bool = False
    """Display FPS counter in live preview"""

    virtual_cam: bool = False
    """Output to virtual camera device"""

    # ======================== Video Output Options ========================
    video_encoder: Optional[str] = None
    """Video encoder (codec) for output"""

    video_quality: Optional[int] = None
    """Video quality parameter (CRF or bitrate)"""

    # ======================== NSFW Filter ========================
    nsfw_filter: bool = False
    """Enable NSFW content filtering"""

    # ======================== UI Toggle State ========================
    # Face processor UI toggles (which enhancers are enabled in UI)
    fp_ui: Dict[str, bool] = field(default_factory=lambda: {
        "face_enhancer": False,
        "face_enhancer_gpen256": False,
        "face_enhancer_gpen512": False,
        "face_enhancer_codeformer": False,
    })
    """UI toggle state for frame processors"""

    # ======================== Threading / Synchronization ========================
    map_lock: threading.RLock = field(default_factory=threading.RLock)
    """Lock protecting source_target_map access across threads"""

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.face_confidence_threshold < 0 or self.face_confidence_threshold > 1:
            raise ValueError("face_confidence_threshold must be between 0 and 1")
        if self.opacity < 0 or self.opacity > 1:
            raise ValueError("opacity must be between 0 and 1")
        if self.codeformer_fidelity < 0 or self.codeformer_fidelity > 1:
            raise ValueError("codeformer_fidelity must be between 0 and 1")
