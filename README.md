<div align="center">

# Fruit Detection & Counting

**Real-time fruit detection, tracking and counting from images, video and live camera feeds.**

<img src="docs/assets/blueberries_counted.gif" width="720" alt="Fruit on a conveyor belt, detected, tracked and counted in real time">

[![CI](https://github.com/felipebridge/fruit-detection-counting/actions/workflows/ci.yml/badge.svg)](https://github.com/felipebridge/fruit-detection-counting/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Ultralytics YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-111F68)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-2A6DB2)](https://mypy-lang.org/)
[![Tested with pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

Combines YOLO11 detection with multi-object tracking so each fruit is counted exactly once, reporting unique fruit rather than per-frame detections.

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
