<h1 align="center">Deep-Live-Cam</h1>

<p align="center">
  Real-time face swap and video deepfake with a single click and only a single image.
</p>

<p align="center">
<a href="https://trendshift.io/repositories/11395" target="_blank"><img src="https://trendshift.io/api/badge/repositories/11395" alt="hacksider%2FDeep-Live-Cam | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <img src="media/demo.gif" alt="Demo GIF" width="800">
</p>

## Disclaimer

This software is intended as a productive tool for the AI-generated media industry, helping artists animate custom characters, create content, and prototype designs.

- **Ethical use**: If using a real person's face, obtain their consent and clearly label output as a deepfake when sharing.
- **Content restrictions**: Built-in checks prevent processing inappropriate media (nudity, graphic content, sensitive material).
- **Legal compliance**: We may shut down the project or add watermarks if legally required.
- **User responsibility**: We are not responsible for end-user actions. Use this software in a manner that respects the rights and dignity of others.

## Quick Start (Pre-built)

<a href="https://deeplivecam.net/index.php/quickstart"><img src="media/Download.png" width="285" height="77" /></a>

Pre-built binaries for Windows and Mac Silicon — no manual installation needed.

## Live Deepfake in 3 Clicks

![easysteps](https://github.com/user-attachments/assets/af825228-852c-411b-b787-ffd9aac72fc6)

1. Select a face
2. Select which camera to use
3. Press live!

## Features

### Mouth Mask

Retain your original mouth for accurate lip movement.

<p align="center">
  <img src="media/ludwig.gif" alt="resizable-gif">
</p>

### Face Mapping

Use different faces on multiple subjects simultaneously.

<p align="center">
  <img src="media/streamers.gif" alt="face_mapping_source">
</p>

### Your Movie, Your Face

Watch movies with any face in real-time.

<p align="center">
  <img src="media/movie.gif" alt="movie">
</p>

### Live Show

Run live shows and performances.

<p align="center">
  <img src="media/live_show.gif" alt="show">
</p>

### Memes

<p align="center">
  <img src="media/meme.gif" alt="show" width="450">
  <br>
  <sub>Created using Many Faces feature in Deep-Live-Cam</sub>
</p>

## Installation

### Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| [git](https://git-scm.com) | Clone the repo | OS package manager |
| [mise](https://mise.jdx.dev) | Manages Python 3.10 with working Tcl/Tk | `curl https://mise.run \| sh` |
| [uv](https://docs.astral.sh/uv/) | Fast Python package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [just](https://github.com/casey/just) | Task runner | `cargo install just` / `brew install just` / [other options](https://github.com/casey/just#installation) |
| [ffmpeg](https://ffmpeg.org) | Video processing | `brew install ffmpeg` / `apt install ffmpeg` / `winget install ffmpeg` |

**GPU drivers** (optional but recommended):
- **NVIDIA**: [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive) + [cuDNN v8.9.7](https://developer.nvidia.com/rdp/cudnn-archive)
- **Apple Silicon**: No extra drivers needed (uses CoreML)
- **AMD**: ROCm drivers

### Setup

```bash
git clone https://github.com/hacksider/Deep-Live-Cam.git
cd Deep-Live-Cam
just setup
```

That's it. `just setup` installs Python dependencies via `uv` and downloads the required models.

### Virtual Camera Setup (Optional)

To output the face-swapped feed as a virtual camera (usable in Zoom, Meet, Discord, etc.):

```bash
just setup-virtualcam
```

This installs the `pyvirtualcam` dependency and prints platform-specific instructions (OBS on macOS/Windows, v4l2loopback on Linux).

## Usage

```bash
just start              # Launch GUI with platform-default GPU (CoreML on macOS, CUDA on Linux/Windows)
just start-cpu          # Launch GUI with CPU only
just start-with rocm    # Launch GUI with a specific execution provider
just start-virtualcam   # Launch GUI with virtual camera output
```

### Image/Video Mode

1. Run `just start`
2. Choose a source face image and a target image/video
3. Click "Start"
4. Output is saved in a directory named after the target video

### Webcam Mode

1. Run `just start`
2. Select a source face image
3. Click "Live"
4. Use OBS to stream the preview, or enable the Virtual Camera toggle

### Headless Mode (No GUI)

```bash
uv run run.py -s source.jpg -t target.mp4 -o output.mp4
```

Providing `-s`/`--source` activates CLI mode — no Tcl/Tk or display required.

## Command Line Arguments

Run `uv run run.py --help` for the full list. To update this section, run `just update-readme`.

<!-- CLI_ARGS_START -->
```
options:
  -h, --help                                               show this help message and exit
  -s SOURCE_PATH, --source SOURCE_PATH                     select a source image
  -t TARGET_PATH, --target TARGET_PATH                     select a target image or video
  -o OUTPUT_PATH, --output OUTPUT_PATH                     select output file or directory
  --frame-processor FRAME_PROCESSOR [FRAME_PROCESSOR ...]  pipeline of frame processors (choices: face_swapper, face_enhancer, face_enhancer_gpen256, face_enhancer_gpen512)
  --keep-fps                                               keep original fps
  --keep-audio                                             keep original audio
  --keep-frames                                            keep temporary frames
  --many-faces                                             process every face
  --nsfw-filter                                            filter NSFW content
  --map-faces                                              map source target faces
  --mouth-mask                                             mask the mouth region
  --video-encoder {libx264,libx265,libvpx-vp9}             adjust output video encoder
  --video-quality [0-51]                                   adjust output video quality
  -l LANG, --lang LANG                                     UI language
  --live-mirror                                            mirror the live camera display
  --live-resizable                                         make the live camera frame resizable
  --virtual-cam                                            output to virtual camera device
  --rife                                                   enable RIFE frame interpolation
  --rife-model {rife-v4.25,rife-v4.25-lite}                RIFE model to use
  --rife-multiplier {2,4}                                  RIFE frame rate multiplier
  --max-memory MAX_MEMORY                                  maximum amount of RAM in GB
  --execution-provider {cpu,...}                            execution provider
  --execution-threads EXECUTION_THREADS                    number of execution threads
  -v, --version                                            show program's version number and exit
```
<!-- CLI_ARGS_END -->

## All `just` Recipes

Run `just` with no arguments to see the full list:

| Recipe | Description |
|--------|-------------|
| `just setup` | Install dependencies + download models |
| `just start` | Run with platform-default GPU |
| `just start-cpu` | Run with CPU only |
| `just start-with <provider>` | Run with a specific provider (cuda, coreml, rocm, directml, openvino, cpu) |
| `just start-virtualcam` | Run with virtual camera output |
| `just setup-virtualcam` | Install virtual camera dependencies |
| `just test` | Run full test suite |
| `just test-quick` | Run fast unit tests |
| `just update-readme` | Regenerate CLI args section from `--help` |
| `just clean` | Remove virtual environment |

## Press

 - [*"Deep-Live-Cam goes viral, allowing anyone to become a digital doppelganger"*](https://arstechnica.com/information-technology/2024/08/new-ai-tool-enables-real-time-face-swapping-on-webcams-raising-fraud-concerns/) - Ars Technica
 - [*"Thanks Deep Live Cam, shapeshifters are among us now"*](https://dataconomy.com/2024/08/15/what-is-deep-live-cam-github-deepfake/) - Dataconomy
 - [*"This free AI tool lets you become anyone during video-calls"*](https://www.newsbytesapp.com/news/science/deep-live-cam-ai-impersonation-tool-goes-viral/story) - NewsBytes
 - [*"OK, this viral AI live stream software is truly terrifying"*](https://www.creativebloq.com/ai/ok-this-viral-ai-live-stream-software-is-truly-terrifying) - Creative Bloq
 - [*"Deepfake AI Tool Lets You Become Anyone in a Video Call With Single Photo"*](https://petapixel.com/2024/08/14/deep-live-cam-deepfake-ai-tool-lets-you-become-anyone-in-a-video-call-with-single-photo-mark-zuckerberg-jd-vance-elon-musk/) - PetaPixel
 - [*"Deep-Live-Cam Uses AI to Transform Your Face in Real-Time, Celebrities Included"*](https://www.techeblog.com/deep-live-cam-ai-transform-face/) - TechEBlog
 - [*"An AI tool that "makes you look like anyone" during a video call is going viral online"*](https://telegrafi.com/en/a-tool-that-makes-you-look-like-anyone-during-a-video-call-is-going-viral-on-the-Internet/) - Telegrafi
 - [*"This Deepfake Tool Turning Images Into Livestreams is Topping the GitHub Charts"*](https://decrypt.co/244565/this-deepfake-tool-turning-images-into-livestreams-is-topping-the-github-charts) - Emerge
 - [*"New Real-Time Face-Swapping AI Allows Anyone to Mimic Famous Faces"*](https://www.digitalmusicnews.com/2024/08/15/face-swapping-ai-real-time-mimic/) - Digital Music News
 - [*"This real-time webcam deepfake tool raises alarms about the future of identity theft"*](https://www.diyphotography.net/this-real-time-webcam-deepfake-tool-raises-alarms-about-the-future-of-identity-theft/) - DIYPhotography
 - [*"That's Crazy, Oh God. That's Fucking Freaky Dude... That's So Wild Dude"*](https://www.youtube.com/watch?time_continue=1074&v=py4Tc-Y8BcY) - SomeOrdinaryGamers
 - [*"Alright look look look, now look chat, we can do any face we want to look like chat"*](https://www.youtube.com/live/mFsCe7AIxq8?feature=shared&t=2686) - IShowSpeed
 - [*"They do a pretty good job matching poses, expression and even the lighting"*](https://www.youtube.com/watch?v=wnCghLjqv3s&t=551s) - TechLinked (LTT)
 - [*"Als Sean Connery an der Redaktionskonferenz teilnahm"*](https://www.golem.de/news/deepfakes-als-sean-connery-an-der-redaktionskonferenz-teilnahm-2408-188172.html) - Golem.de (German)
 - [*"What the F***! Why do I look like Vinny Jr? I look exactly like Vinny Jr!? No, this shit is crazy! Bro This is F*** Crazy! "*](https://youtu.be/JbUPRmXRUtE?t=3964) - IShowSpeed

## Credits

- [ffmpeg](https://ffmpeg.org/): for making video-related operations easy
- [Henry](https://github.com/henryruhs): One of the major contributors in this repo
- [deepinsight](https://github.com/deepinsight): for their [insightface](https://github.com/deepinsight/insightface) project which provided a well-made library and models. Please be reminded that the [use of the model is for non-commercial research purposes only](https://github.com/deepinsight/insightface?tab=readme-ov-file#license).
- [havok2-htwo](https://github.com/havok2-htwo): for sharing the code for webcam
- [GosuDRM](https://github.com/GosuDRM): for the open version of roop
- [pereiraroland26](https://github.com/pereiraroland26): Multiple faces support
- [vic4key](https://github.com/vic4key): For supporting/contributing to this project
- [kier007](https://github.com/kier007): for improving the user experience
- [qitianai](https://github.com/qitianai): for multi-lingual support
- [laurigates](https://github.com/laurigates): Decoupling stuffs to make everything faster!
- and [all developers](https://github.com/hacksider/Deep-Live-Cam/graphs/contributors) behind libraries used in this project.
- Footnote: Please be informed that the base author of the code is [s0md3v](https://github.com/s0md3v/roop)
- All the wonderful users who helped make this project go viral by starring the repo

[![Stargazers](https://reporoster.com/stars/hacksider/Deep-Live-Cam)](https://github.com/hacksider/Deep-Live-Cam/stargazers)

## Contributions

![Alt](https://repobeats.axiom.co/api/embed/fec8e29c45dfdb9c5916f3a7830e1249308d20e1.svg "Repobeats analytics image")

<a href="https://star-history.com/#hacksider/deep-live-cam&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date" />
 </picture>
</a>
