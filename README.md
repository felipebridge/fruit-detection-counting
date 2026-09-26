<div align="center">

# Fruit Detection & Counting

**Detect fruit in images and video with YOLO11, and count each one exactly once.**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Ultralytics YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-111F68)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)

<br>

<img src="docs/assets/video_counting.gif" height="280" alt="Camera pan over a fruit bowl with persistent track ids and a running unique count">
&nbsp;
<img src="docs/assets/image_detection.jpg" height="280" alt="Fruit bowl with bounding boxes and a per-class count panel">

<sub>Video: every fruit keeps one id and is counted once &nbsp;·&nbsp; Image: per-class count</sub>

</div>

<br>

## Features

- Images, image folders, video files and webcams through one command
- Unique counting in video: a fruit seen in 100 frames counts once, not 100 times
- Annotated image or video, `summary.json` and per-frame `frames.csv`
- Counting error against a ground-truth CSV (`--ground-truth`)
- Any Ultralytics detector via `--weights`

## Installation

```bash
git clone https://github.com/felipebridge/fruit-detection-counting.git
cd fruit-detection-counting
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

The YOLO11n weights (~5 MB) download on first run. For a GPU, install the matching
[PyTorch build](https://pytorch.org/get-started/locally/) first.

## Usage

```bash
fruit-counter -i data/samples/fruit_bowl.jpg        # image
fruit-counter -i path/to/images/                    # folder
fruit-counter -i 0 --show                           # webcam (q to stop)

python scripts/make_demo_video.py                   # build the demo clip, then:
fruit-counter -i data/samples/fruit_bowl_pan.mp4
```

```text
Frames processed:      150
Unique fruits counted: 8
  - apple: 5
  - orange: 3
Outputs: outputs/fruit_bowl_pan_20260926-133041
```

Common flags: `--conf`, `--device cuda:0`, `--stride 2`, `--weights`, `--classes`.
All settings live in [`configs/default.yaml`](configs/default.yaml); see `fruit-counter --help`.

```python
from fruit_counter import FruitCountingRunner, load_config, resolve_source

report = FruitCountingRunner(load_config()).run(resolve_source("orchard.mp4"))
print(report.fruit_count, report.results["counts_by_class"])
```

## How it works

1. **Detect** — YOLO with class-agnostic NMS, so one fruit is never boxed under two labels.
2. **Track** — constant-velocity box prediction, matched to detections by IoU (Hungarian algorithm).
3. **Count** — a track counts after `min_hits` consecutive matches and survives `max_age`
   missed frames; its class is a confidence-weighted vote.

On the sample clip, 657 per-frame detections collapse to 8 tracks, none counted twice
(the bowl holds 9 fruits, labelled by hand).

## Limitations

- The default COCO model knows only **apple, banana and orange**; lemons and limes are
  mislabelled or missed. For other fruit, train a detector and pass `--weights`.
- A fruit that leaves the view longer than `max_age` is counted again on return.
- A fruit that is never detected is never counted.

## Development

```bash
pip install -e ".[dev]"
pytest && ruff check . && mypy
```

<sub>No license chosen yet. Detection uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
Sample image is CC0 ([attribution](data/samples/ATTRIBUTION.md)).</sub>
