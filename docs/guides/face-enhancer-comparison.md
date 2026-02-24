# Face Enhancer Comparison Guide

Deep-Live-Cam supports three face enhancement models. Each offers different quality/performance trade-offs for real-time and offline processing.

## Available Enhancers

| Model | Input Size | Output Size | Quality | Speed | Best For |
|-------|-----------|-------------|---------|-------|----------|
| GFPGAN | 512x512 | 1024x1024 | Highest | Slowest (~60-150ms/face) | Offline video processing, single-face |
| GPEN-512 | 512x512 | 512x512 | High | Medium (~30-80ms/face) | Balanced live/offline use |
| GPEN-256 | 256x256 | 256x256 | Good | Fastest (~10-30ms/face) | Real-time webcam, multi-face |

## Enabling Enhancers

### GUI
Toggle enhancers in the UI panel:
- **Face Enhancer** -- GFPGAN (default)
- **Face Enhancer GPEN-256** -- Lightweight, fastest
- **Face Enhancer GPEN-512** -- Medium quality/speed

### CLI
```bash
uv run run.py --frame-processor face_swapper face_enhancer
uv run run.py --frame-processor face_swapper face_enhancer_gpen256
uv run run.py --frame-processor face_swapper face_enhancer_gpen512
```

## Performance by Hardware

### Apple Silicon (M1/M2/M3/M4)
| Model | CoreML FPS (1 face) | CoreML FPS (3 faces) |
|-------|--------------------|--------------------|
| No enhancer | 20-25 | 15-20 |
| GPEN-256 | 15-20 | 10-15 |
| GPEN-512 | 10-15 | 5-10 |
| GFPGAN | 5-8 | 2-4 |

### NVIDIA GPU (RTX 3060+)
| Model | CUDA FPS (1 face) | CUDA FPS (3 faces) |
|-------|-------------------|--------------------|
| No enhancer | 25-30 | 20-25 |
| GPEN-256 | 20-25 | 15-20 |
| GPEN-512 | 15-20 | 8-12 |
| GFPGAN | 8-12 | 3-6 |

*FPS values are approximate and vary by system configuration.*

## Quality Comparison

### GFPGAN (Highest Quality)
- Best detail restoration (eyes, teeth, skin texture)
- Upscales from 512 to 1024 internally (2x super-resolution)
- May over-smooth in some cases
- Recommended for: final video output, photo enhancement

### GPEN-512 (Balanced)
- Good detail preservation without super-resolution
- Handles diverse face angles well
- Lower VRAM usage than GFPGAN
- Recommended for: live webcam with good hardware, balanced workflows

### GPEN-256 (Fastest)
- Lightweight model, minimal quality enhancement
- Best for maintaining high FPS in real-time scenarios
- Works well at webcam resolution (480p-720p)
- Recommended for: multi-face live mode, low-end hardware, mobile demos

## Optimization Tips

1. **Skip-frame enhancement**: Set `enhancer_skip_interval` > 1 to enhance every Nth frame
2. **Half-rate processing**: Enable for additional FPS boost (processes keyframes only)
3. **Combine strategies**: GPEN-256 + skip-frame + half-rate for maximum FPS on multi-face scenes
4. **Single enhancer only**: Only one enhancer should be active at a time
