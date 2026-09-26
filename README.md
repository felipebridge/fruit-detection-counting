# Fruit Detection & Counting

Detects fruit in images and video with YOLO11 and counts every fruit **once**, even
when it stays in view for hundreds of frames.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Ultralytics YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-111F68)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)

| Image: per-class count | Video: tracked, each fruit counted once |
|:---:|:---:|
| <img src="docs/assets/image_detection.jpg" width="330" alt="Fruit bowl with bounding boxes and a count panel"> | <img src="docs/assets/video_counting.gif" width="480" alt="Camera pan over a fruit bowl with persistent track ids and a running unique count"> |

## Features

- **Images, folders, videos and webcams** with a single command
- **Unique counting in video.** A built-in IoU tracker with motion prediction gives
  each fruit one id, so a fruit seen in 100 frames counts once, not 100 times.
- **Useful outputs:** annotated image or video, `summary.json` and per-frame `frames.csv`
- **Counting error** against a ground-truth CSV (`--ground-truth`)
- **Custom models:** any Ultralytics detector via `--weights`

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
fruit-counter -i 0 --show                           # webcam, live preview (q to stop)

python scripts/make_demo_video.py                   # build the demo clip, then:
fruit-counter -i data/samples/fruit_bowl_pan.mp4
```

```text
Source: data/samples/fruit_bowl_pan.mp4 (video)
Frames processed:      150
Unique fruits counted: 8
Max visible in frame:  8
Processing speed:      12.0 fps
  - apple: 5
  - orange: 3
Outputs: outputs/fruit_bowl_pan_20260926-133041
```

Useful flags: `--conf`, `--device cuda:0`, `--stride 2`, `--weights`, `--classes`.
Every setting is documented in [`configs/default.yaml`](configs/default.yaml) (`-c`
to load your own); run `fruit-counter --help` for the full list.

From Python:

```python
from fruit_counter import FruitCountingRunner, load_config, resolve_source

report = FruitCountingRunner(load_config()).run(resolve_source("orchard.mp4"))
print(report.fruit_count, report.results["counts_by_class"])
```

## How it works

1. **Detect.** YOLO runs with class-agnostic NMS, so one fruit can't be boxed as both
   "apple" and "orange".
2. **Track.** Boxes are predicted forward with a constant-velocity model and matched
   to detections by IoU (Hungarian algorithm). Weak detections can extend a track
   but never start one.
3. **Count.** A track is counted after `min_hits` consecutive matches, which filters
   one-frame false positives, and survives `max_age` missed frames through
   occlusions. Its class is a confidence-weighted vote over its lifetime.

Inference dominates the runtime: roughly 70–80 ms per 800×450 frame for YOLO11n on
a laptop CPU, plus ~4 ms for tracking, drawing and encoding.

## Results on the sample

The CC0 sample bowl holds 9 fruits, labelled by hand: 3 oranges, 1 apple, 2 lemons and 3 limes.
This is one example, not a benchmark.

| Input | Counted | True |
|---|:---:|:---:|
| `fruit_bowl.jpg` | 6 | 9 |
| `fruit_bowl_pan.mp4` (150 frames) | 8 | 9 |

In the video, 657 detections summed over frames reduce to 8 tracks. A frame-by-frame
check shows 8 distinct fruits, none counted twice. Lemons and limes are not COCO classes,
so they are labelled "apple" or missed.

## Limitations

- The default COCO model only knows **apple, banana and orange**. For other fruit,
  train a detector with Ultralytics and pass `--weights my_model.pt --classes`.
- A fruit that leaves the view, or stays hidden longer than `max_age`, is counted
  again on return. There is no appearance-based re-identification.
- A fruit that is never detected is never counted.

## Development

```bash
pip install -e ".[dev]"
pytest                                   # includes one real-inference test
ruff check . && ruff format --check . && mypy
```

## License

No license has been chosen for this project's code yet. Detection uses
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0). The sample
image is CC0 ([attribution](data/samples/ATTRIBUTION.md)).
