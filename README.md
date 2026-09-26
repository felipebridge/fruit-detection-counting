<div align="center">

# Fruit Detection & Counting

**Point a camera at fruit. Get a count where each fruit counts exactly once.**

[![CI](https://github.com/felipebridge/fruit-detection-counting/actions/workflows/ci.yml/badge.svg)](https://github.com/felipebridge/fruit-detection-counting/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Ultralytics YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-111F68)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

<br>

<img src="docs/assets/video_counting.gif" height="280" alt="Camera pan over a fruit bowl with persistent track ids and a running unique count">
&nbsp;
<img src="docs/assets/image_detection.jpg" height="280" alt="Fruit bowl with bounding boxes and a per-class count panel">

</div>

<br>

A detector tells you what is in a frame. Across a video, the same apple shows up in
hundreds of frames. This project tracks every fruit, so the total is the number of
fruit, not the number of detections.

## Features

- **One command** for images, image folders, video files and webcams
- **Unique counting**: each fruit keeps one track id for as long as it stays in view
- **Ready-to-use outputs**: annotated media, `summary.json` and per-frame `frames.csv`
- **Built-in evaluation**: counting error against your ground-truth CSV
- **Bring your own model**: any Ultralytics detector through `--weights`

## Quick Start

```bash
git clone https://github.com/felipebridge/fruit-detection-counting.git
cd fruit-detection-counting
pip install -e .

fruit-counter -i data/samples/fruit_bowl.jpg
```

The YOLO11n weights (~5 MB) download automatically on first run.

## Usage

```bash
fruit-counter -i path/to/images/                  # folder of images
fruit-counter -i orchard.mp4 --device cuda:0      # video, on a GPU
fruit-counter -i 0 --show                         # webcam, live preview

python scripts/make_demo_video.py                 # build the sample clip, then:
fruit-counter -i data/samples/fruit_bowl_pan.mp4
```

```text
Frames processed:      150
Unique fruits counted: 8
  - apple: 5
  - orange: 3
Outputs: outputs/fruit_bowl_pan_20260926-133041
```

The default COCO model detects apples, bananas and oranges. For other fruit, pass your
own weights with `--weights my_model.pt --classes ...`. All settings live in
[`configs/default.yaml`](configs/default.yaml); run `fruit-counter --help` for every flag.

From Python:

```python
from fruit_counter import FruitCountingRunner, load_config, resolve_source

report = FruitCountingRunner(load_config()).run(resolve_source("orchard.mp4"))
print(report.fruit_count, report.results["counts_by_class"])
```

## How It Works

```text
frame ─▶ Detect ─▶ Track ─▶ Count ─▶ annotated media + summary.json
```

1. **Detect**: YOLO11 with class-agnostic NMS, so one fruit never gets two labels.
2. **Track**: boxes move forward with a constant-velocity model and are matched to new
   detections by IoU with the Hungarian algorithm.
3. **Count**: a track counts once it has been confirmed over several frames. Its label is
   a confidence-weighted vote over its lifetime.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and checks.

## License

[MIT](LICENSE) © felipebridge. Detection is powered by
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0). The sample
image is CC0 ([attribution](data/samples/ATTRIBUTION.md)).
