<div align="center">

# Fruit Detection & Counting

**Point a camera at fruit. Get a count where each fruit counts exactly once.**

<img src="docs/assets/blueberries_counted.gif" width="720" alt="Blueberries on a conveyor belt, each one tracked and counted once">

[![CI](https://github.com/felipebridge/fruit-detection-counting/actions/workflows/ci.yml/badge.svg)](https://github.com/felipebridge/fruit-detection-counting/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Ultralytics YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-111F68)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

YOLO11 detection plus tracking, so the total is the number of fruit, not the number of detections.

## Quick Start

```bash
git clone https://github.com/felipebridge/fruit-detection-counting.git
cd fruit-detection-counting
pip install -e .
fruit-counter            # processes every video in videos_to_processing/
```

Results go to `outputs/<video>_<timestamp>/`: an annotated H.264 video, `summary.json` and `frames.csv`.
Install [ffmpeg](https://ffmpeg.org/download.html) for H.264 output with audio.

```bash
fruit-counter -i fruit_bowl.jpg                     # image or folder
fruit-counter -i orchard.mp4 --device cuda:0        # video on a GPU
fruit-counter -i 0 --show                           # webcam
fruit-counter -i video.mp4 --weights my_model.pt    # your own model
```

Settings live in [`configs/default.yaml`](configs/default.yaml); see `fruit-counter --help`.

## How It Works

**Detect** (YOLO11, class-agnostic NMS) → **Track** (constant-velocity + Hungarian IoU matching) → **Count** (once per confirmed track, label by confidence-weighted vote) → **Render**.

## License

[MIT](LICENSE) © felipebridge. Powered by [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0). Contributions welcome, see [CONTRIBUTING.md](CONTRIBUTING.md).
