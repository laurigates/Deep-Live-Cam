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

This deepfake software is designed to be a productive tool for the AI-generated media industry. It can assist artists in animating custom characters, creating engaging content, and even using models for clothing design.

We are aware of the potential for unethical applications and are committed to preventative measures. A built-in check prevents the program from processing inappropriate media (nudity, graphic content, sensitive material like war footage, etc.). We will continue to develop this project responsibly, adhering to the law and ethics. We may shut down the project or add watermarks if legally required.

- **Ethical Use**: Users are expected to use this software responsibly and legally. If using a real person's face, obtain their consent and clearly label any output as a deepfake when sharing online.
- **Content Restrictions**: The software includes built-in checks to prevent processing inappropriate media, such as nudity, graphic content, or sensitive material.
- **Legal Compliance**: We adhere to all relevant laws and ethical guidelines. If legally required, we may shut down the project or add watermarks to the output.
- **User Responsibility**: We are not responsible for end-user actions. Users must ensure their use of the software aligns with ethical standards and legal requirements.

By using this software, you agree to these terms and commit to using it in a manner that respects the rights and dignity of others.

## Quick Demo

![easysteps](https://github.com/user-attachments/assets/af825228-852c-411b-b787-ffd9aac72fc6)

1. Select a face
2. Select which camera to use
3. Press live!

## Features

### Mouth Mask

**Retain your original mouth for accurate movement using Mouth Mask**

<p align="center">
  <img src="media/ludwig.gif" alt="resizable-gif">
</p>

### Face Mapping

**Use different faces on multiple subjects simultaneously**

<p align="center">
  <img src="media/streamers.gif" alt="face_mapping_source">
</p>

### Your Movie, Your Face

**Watch movies with any face in real-time**

<p align="center">
  <img src="media/movie.gif" alt="movie">
</p>

### Live Show

**Run Live shows and performances**

<p align="center">
  <img src="media/live_show.gif" alt="show">
</p>

### Memes

**Create Your Most Viral Meme Yet**

<p align="center">
  <img src="media/meme.gif" alt="show" width="450">
  <br>
  <sub>Created using Many Faces feature in Deep-Live-Cam</sub>
</p>

## Installation

### Prerequisites

- **Python 3.10** (managed via [mise](https://mise.jdx.dev/) or installed manually)
- **ffmpeg** on PATH
- **git**

### Setup

```bash
git clone https://github.com/hacksider/Deep-Live-Cam.git
cd Deep-Live-Cam
just setup    # installs dependencies (uv sync) + downloads models
```

> If you don't have `just`, install it via `brew install just`, `cargo install just`, or see [just installation](https://github.com/casey/just#installation). Alternatively, run `uv sync` and download models manually into `models/`.

### GPU Providers

| Platform | Provider | How to run |
|----------|----------|------------|
| macOS ARM (M1/M2/M3/M4) | CoreML | `just start` (auto-detected) |
| NVIDIA | CUDA | `just start` (auto-detected) |
| AMD | ROCm | `just start-with rocm` |
| Intel | OpenVINO | `just start-with openvino` |
| Windows (any GPU) | DirectML | `just start-with directml` |
| CPU only | - | `just start-cpu` |

GPU-specific ONNX runtimes are handled automatically by `pyproject.toml` platform markers. For CUDA, ensure [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) are installed.

## Usage

### GUI (Live Webcam)

```bash
just start              # launch GUI with platform-default GPU
```

Select a source face image, choose your camera, and click **Live**.

### Headless (Image/Video)

```bash
uv run run.py -s source.jpg -t target.mp4 -o output.mp4
uv run run.py -s source.jpg -t target.jpg -o output.jpg --frame-processor face_swapper face_enhancer
```

### Virtual Camera

Send processed output directly to Zoom, Google Meet, Discord — no screen capture needed.

```bash
just start-virtualcam
```

**One-time setup**: macOS — install OBS 30+, start/stop virtual camera once. Linux — `sudo apt install v4l2loopback-dkms && sudo modprobe v4l2loopback devices=1`. Windows — install OBS 26+.

## Command Line Arguments

<!-- CLI_ARGS_START -->
*Deep-Live-Cam v2.0.3c* — run `uv run run.py --help` for latest options.

| Flag | Description | Default | Choices |
|------|-------------|---------|---------|
| `-s, --source` | select an source image |  |  |
| `-t, --target` | select an target image or video |  |  |
| `-o, --output` | select output file or directory |  |  |
| `--frame-processor` | pipeline of frame processors | `'face_swapper'` | `'face_swapper', 'face_enhancer', 'face_enhancer_gpen256', 'face_enhancer_gpen512'` |
| `--keep-fps` | keep original fps | `False` |  |
| `--keep-audio` | keep original audio | `True` |  |
| `--keep-frames` | keep temporary frames | `False` |  |
| `--many-faces` | process every face | `False` |  |
| `--nsfw-filter` | filter the NSFW image or video | `False` |  |
| `--map-faces` | map source target faces | `False` |  |
| `--mouth-mask` | mask the mouth region | `False` |  |
| `--video-encoder` | adjust output video encoder | `'libx264'` | `'libx264', 'libx265', 'libvpx-vp9'` |
| `--video-quality` | adjust output video quality | `18` | `0-51` |
| `-l, --lang` | Ui language | `'en'` |  |
| `--live-mirror` | The live camera display as you see it in the front-facing camera frame | `False` |  |
| `--live-resizable` | The live camera frame is resizable | `False` |  |
| `--virtual-cam` | output to virtual camera device | `False` |  |
| `--rife` | enable RIFE frame interpolation for video output | `False` |  |
| `--rife-model` | RIFE model to use | `'rife-v4.25-lite'` | `'rife-v4.25', 'rife-v4.25-lite'` |
| `--rife-multiplier` | RIFE frame rate multiplier (2=double, 4=quadruple) | `2` | `2, 4` |
| `--half-rate` | enable half-rate face processing with RIFE interpolation for live mode | `False` |  |
| `--keyframe-interval` | process every Nth frame in half-rate mode (2-10) | `2` | `2-10` |
| `--max-memory` | maximum amount of RAM in GB | `(auto)` |  |
| `--execution-provider` | execution provider | `'cpu'` | `(auto)` |
| `--execution-threads` | number of execution threads | `(auto)` |  |
<!-- CLI_ARGS_END -->

Using `-s`/`--source` triggers headless (CLI) mode — no GUI window opens.

## Development

```bash
just --list             # see all available recipes
just test-quick         # fast unit tests
just test               # full test suite
just test-cov           # tests with coverage report
```

See [CLAUDE.md](CLAUDE.md) for full developer documentation.

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
- [Henry](https://github.com/henryruhs): One of the major contributor in this repo
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

## Stars

<a href="https://star-history.com/#hacksider/deep-live-cam&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=hacksider/deep-live-cam&type=Date" />
 </picture>
</a>
