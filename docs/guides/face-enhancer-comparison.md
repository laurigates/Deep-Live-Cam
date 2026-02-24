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
- **Face Enhancer** -- GFPGAN (highest quality, slowest)
- **Face Enhancer GPEN-256** -- Lightweight, fastest
- **Face Enhancer GPEN-512** -- Balanced quality/speed

All three enhancers now work correctly in live mode and offline video/image processing. You can toggle them on or off independently. The UI automatically keeps the frame processors pipeline in sync with your selections.

**Note:** Only one enhancer should be active at a time for best results. Enabling multiple enhancers will stack them in sequence, which adds minimal quality improvement but significantly increases processing time.

### CLI
```bash
# Video/image processing (offline)
uv run run.py -s source.jpg -t target.mp4 -o output.mp4 --frame-processor face_swapper face_enhancer
uv run run.py -s source.jpg -t target.jpg -o output.jpg --frame-processor face_swapper face_enhancer_gpen512

# Live webcam (with live mode optimization)
just start  # Launches GUI; toggle enhancers there
uv run run.py --virtual-cam --frame-processor face_swapper face_enhancer_gpen256 --live-enhance-size 192
```

## Performance by Hardware

All FPS values below use default settings (native resolution, no `--live-enhance-size` optimization). With live-enhance-size tuning, you can expect 15-40% higher FPS depending on the optimization level.

### Apple Silicon (M1/M2/M3/M4) — CoreML

| Model | 1 Face (FPS) | 3 Faces (FPS) | Notes |
|-------|--------|---------|-------|
| No enhancer | 20-25 | 15-20 | Baseline |
| GPEN-256 | 15-20 | 10-15 | Fast, minimal enhancement |
| GPEN-256 (128 align) | 18-23 | 13-18 | With `--live-enhance-size 128` |
| GPEN-512 | 10-15 | 5-10 | Balanced quality/speed |
| GPEN-512 (256 align) | 14-20 | 8-14 | With `--live-enhance-size 256` |
| GFPGAN | 5-8 | 2-4 | Highest quality |

### NVIDIA GPU (RTX 3060+) — CUDA

| Model | 1 Face (FPS) | 3 Faces (FPS) | Notes |
|-------|--------|---------|-------|
| No enhancer | 25-30 | 20-25 | Baseline |
| GPEN-256 | 20-25 | 15-20 | Fast, minimal enhancement |
| GPEN-256 (128 align) | 23-28 | 18-24 | With `--live-enhance-size 128` |
| GPEN-512 | 15-20 | 8-12 | Balanced quality/speed |
| GPEN-512 (256 align) | 19-25 | 12-18 | With `--live-enhance-size 256` |
| GFPGAN | 8-12 | 3-6 | Highest quality |

*FPS values are approximate and vary by system configuration, face size, and frame resolution.*

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

## Live Mode Optimization

In live webcam mode, face enhancers can be optimized for speed without sacrificing quality:

### Low-Resolution Face Alignment (`--live-enhance-size`)

By default, enhancers align and process faces at their native resolution (512 for GPEN-512 and GFPGAN, 256 for GPEN-256). For live mode, Deep-Live-Cam supports reducing the alignment/paste resolution to trade some quality for speed.

**How it works:**
1. Face is aligned (warpAffine) at `live-enhance-size` instead of the model's native size — cheaper
2. The cropped face is upscaled to the model's expected input size before inference (model always runs at full resolution)
3. The enhanced output is downscaled back to `live-enhance-size` (bicubic) before paste-back
4. The inverse affine paste operates on the smaller buffer — cheaper

The model inference cost is unchanged; savings come from the warp/mask/paste operations which scale as O(size²). At size 256 vs 512 that's a 4× reduction in those steps.

**Configuration:**
```bash
# Pass via uv (just recipes don't accept extra flags)
uv run run.py --execution-provider coreml --live-enhance-size 128   # Smallest, fastest
uv run run.py --execution-provider coreml --live-enhance-size 256   # Default
uv run run.py --execution-provider coreml --live-enhance-size 384   # Slightly higher quality
uv run run.py --execution-provider coreml --live-enhance-size 512   # Native resolution (no optimization)
```

**Recommended settings by hardware:**
| Hardware | GPEN-256 | GPEN-512 | GFPGAN |
|----------|----------|----------|--------|
| Apple Silicon (M1-M4) | 128 or 192 | 256 | 384 |
| NVIDIA RTX 3060+ | 192 or 256 | 384 | 512 |
| Low-end GPU | 128 | 192 | 256 |

## Optimization Tips

1. **Live-enhance-size tuning**: Start at 256 (default) and reduce to 128 if FPS is insufficient
2. **Skip-frame enhancement**: Set `enhancer_skip_interval` > 1 to enhance every Nth frame
3. **Half-rate processing**: Enable for additional FPS boost (processes keyframes only)
4. **Combine strategies**: GPEN-256 + 128 alignment + skip-frame + half-rate for maximum FPS on multi-face scenes
5. **Single enhancer only**: Only one enhancer should be active at a time; toggle via UI or ensure only one `--frame-processor` flag is used
