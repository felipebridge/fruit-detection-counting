import numpy as np

from conftest import ScriptedDetector, make_detection
from fruit_counter.config import TrackingConfig
from fruit_counter.detector import Detection
from fruit_counter.pipeline import VideoPipeline
from fruit_counter.visualization import FRUIT_COLORS, BoxSmoother


def fruit_passing(frames: int, y: float, class_name: str = "apple", start: int = 0):
    """Detections of one fruit moving right at 6 px/frame, visible from ``start``."""
    return {
        f: make_detection(6 * f, y, 6 * f + 40, y + 40, 0.9, class_name)
        for f in range(start, start + frames)
    }


def build_script(*fruits: dict[int, Detection], length: int) -> list[list[Detection]]:
    return [[fruit[f] for fruit in fruits if f in fruit] for f in range(length)]


def test_unique_count_vs_visible_count(blank_frame: np.ndarray) -> None:
    script = build_script(
        fruit_passing(30, y=10),
        fruit_passing(30, y=80, class_name="orange", start=10),
        length=40,
    )
    pipeline = VideoPipeline(ScriptedDetector(script), TrackingConfig(min_hits=3))
    results = [pipeline.process_frame(blank_frame, i, i / 10) for i in range(40)]

    # Frame 15: both fruits visible and confirmed.
    assert results[15].visible_count == 2
    assert results[15].tracked_count == 2
    assert results[15].unique_count == 2
    # Frames 30-39: only the orange remains visible, but the apple stays counted.
    assert results[35].visible_count == 1
    assert results[35].unique_count == 2
    # Each fruit is reported as new exactly once, when its track is confirmed.
    newly = [(r.frame_index, r.newly_counted) for r in results if r.newly_counted]
    assert newly == [(2, [1]), (12, [2])]

    summary = pipeline.summary()
    assert summary["frames_processed"] == 40
    assert summary["unique_fruit_count"] == 2
    assert summary["counts_by_class"] == {"apple": 1, "orange": 1}
    assert summary["detections_summed_over_frames"] == 60  # the naive per-frame total
    assert summary["max_visible_in_frame"] == 2
    assert summary["mean_visible_per_frame"] == 60 / 40
    assert [obj["track_id"] for obj in summary["objects"]] == [1, 2]


def test_frame_rows_are_flat_and_serialisable(blank_frame: np.ndarray) -> None:
    script = build_script(fruit_passing(5, y=10), length=5)
    pipeline = VideoPipeline(ScriptedDetector(script), TrackingConfig(min_hits=3))
    rows = [pipeline.process_frame(blank_frame, i, i / 25).to_row() for i in range(5)]
    assert rows[2] == {
        "frame_index": 2,
        "timestamp_s": 0.08,
        "visible_count": 1,
        "tracked_count": 1,
        "unique_count": 1,
        "new_ids": "1",
        "inference_ms": rows[2]["inference_ms"],
    }
    assert rows[1]["tracked_count"] == 0  # still tentative


def test_frame_index_defaults_to_running_count(blank_frame: np.ndarray) -> None:
    pipeline = VideoPipeline(ScriptedDetector([[], []]))
    assert [pipeline.process_frame(blank_frame).frame_index for _ in range(2)] == [0, 1]


def test_empty_stream_summary() -> None:
    summary = VideoPipeline(ScriptedDetector([])).summary()
    assert summary["unique_fruit_count"] == 0
    assert summary["mean_visible_per_frame"] == 0.0
    assert summary["objects"] == []


def test_annotated_frame_shows_confirmed_ids(blank_frame: np.ndarray) -> None:
    script = build_script(fruit_passing(3, y=100), length=3)
    pipeline = VideoPipeline(ScriptedDetector(script), TrackingConfig(min_hits=3))
    for i in range(3):
        result = pipeline.process_frame(blank_frame, i)
    annotated = pipeline.annotate(blank_frame, result)
    assert annotated is blank_frame  # drawn in place
    assert annotated.any()


def test_tentative_detections_are_not_drawn(blank_frame: np.ndarray) -> None:
    script = build_script(fruit_passing(1, y=100), length=1)
    pipeline = VideoPipeline(ScriptedDetector(script), TrackingConfig(min_hits=3))
    result = pipeline.process_frame(blank_frame, 0)
    annotated = pipeline.annotate(blank_frame, result)
    assert not annotated[150:, :].any()  # nothing is drawn below the count card


def test_counted_fruit_stays_drawn_through_a_missed_frame(blank_frame: np.ndarray) -> None:
    script = build_script(fruit_passing(4, y=180), length=6)  # detected in frames 0-3 only
    pipeline = VideoPipeline(ScriptedDetector(script), TrackingConfig(min_hits=3))
    for i in range(5):
        frame = blank_frame.copy()
        annotated = pipeline.annotate(frame, pipeline.process_frame(frame, i))
    # Frame 4 has no detection, yet the apple is still drawn near its last position.
    apple = np.array(FRUIT_COLORS["apple"])
    region = annotated[170:230, 10:90].reshape(-1, 3).astype(int)
    assert (np.abs(region - apple).max(axis=1) < 40).any()


def test_box_smoother_steadies_jitter_and_drops_lost_tracks() -> None:
    smoother = BoxSmoother(alpha=0.5, hold_frames=2)
    assert smoother.update({1: (0, 0, 10, 10)}) == {1: (0, 0, 10, 10)}
    steadied = smoother.update({1: (4, 0, 14, 10)})[1]
    assert 0 < steadied[0] < 4  # moves toward the detection, not all the way
    assert 1 in smoother.update({})  # coasts through missed frames
    assert 1 in smoother.update({})
    assert smoother.update({}) == {}  # dropped after hold_frames misses
