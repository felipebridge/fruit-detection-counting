import numpy as np
import pytest

from conftest import make_detection
from fruit_counter.config import TrackingConfig
from fruit_counter.detector import Detection
from fruit_counter.tracker import FruitTracker, iou_matrix, match_by_iou


def moving_box(
    frame: int,
    x0: float = 10,
    speed: float = 8,
    y: float = 50,
    size: float = 40,
    conf: float = 0.9,
    class_name: str = "apple",
) -> Detection:
    """A box of ``size`` px moving horizontally at ``speed`` px/frame."""
    x = x0 + speed * frame
    return make_detection(x, y, x + size, y + size, conf, class_name)


def iou(a: Detection, b: Detection) -> float:
    return float(iou_matrix(np.array([a.box]), np.array([b.box]))[0, 0])


def ids_over(tracker: FruitTracker, frames: list[list[Detection]]) -> list[list[int | None]]:
    return [[t.track_id for t in tracker.update(dets)] for dets in frames]


# ----------------------------------------------------------------- matching


def test_iou_matrix_known_values() -> None:
    a = np.array([[0, 0, 10, 10], [100, 100, 110, 110]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [200, 200, 210, 210]])
    matrix = iou_matrix(a, b)
    assert matrix.shape == (2, 3)
    np.testing.assert_allclose(matrix[0], [1.0, 50 / 150, 0.0])
    np.testing.assert_allclose(matrix[1], [0.0, 0.0, 0.0])


def test_iou_matrix_handles_empty_inputs_and_degenerate_boxes() -> None:
    assert iou_matrix(np.empty((0, 4)), np.array([[0, 0, 1, 1]])).shape == (0, 1)
    assert iou_matrix(np.array([[0, 0, 1, 1]]), np.empty((0, 4))).shape == (1, 0)
    zero_area = np.array([[5, 5, 5, 5]])
    assert iou_matrix(zero_area, zero_area)[0, 0] == 0.0


def test_hungarian_matching_is_globally_optimal() -> None:
    # Greedy matching would pair track 0 with detection 0 (0.6) and leave track 1 with
    # nothing useful; the optimal assignment is 0->1 and 1->0.
    iou = np.array([[0.6, 0.5], [0.55, 0.0]])
    matches, unmatched = match_by_iou(iou, 0.3)
    assert sorted(matches) == [(0, 1), (1, 0)]
    assert unmatched == []


def test_matching_respects_min_iou() -> None:
    assert match_by_iou(np.array([[0.1]]), 0.3) == ([], [0])


# ----------------------------------------------------------------- tracker


def test_track_is_confirmed_after_min_hits() -> None:
    tracker = FruitTracker(TrackingConfig(min_hits=3))
    ids = ids_over(tracker, [[moving_box(f)] for f in range(4)])
    assert ids == [[None], [None], [1], [1]]
    assert tracker.confirmed_count == 1


def test_min_hits_of_one_confirms_immediately() -> None:
    tracker = FruitTracker(TrackingConfig(min_hits=1))
    assert ids_over(tracker, [[moving_box(0)]]) == [[1]]


def test_object_seen_for_100_frames_keeps_a_single_identity() -> None:
    tracker = FruitTracker()
    ids = ids_over(tracker, [[moving_box(f, speed=3)] for f in range(100)])
    assert {i for frame in ids for i in frame if i is not None} == {1}
    assert tracker.confirmed_count == 1


def test_fast_motion_is_followed_thanks_to_velocity_prediction() -> None:
    # A 40 px fruit accelerating to 30 px/frame (e.g. a camera pan speeding up).
    # At top speed consecutive boxes overlap with IoU ~0.14, below match_iou=0.3, so
    # matching raw previous positions would break the track every frame.
    speeds = [min(6 + 3 * f, 30) for f in range(25)]
    xs = np.cumsum([0, *speeds])
    frames = [[make_detection(x, 50, x + 40, 90)] for x in xs]
    last_step = iou(make_detection(xs[-2], 50, xs[-2] + 40, 90), frames[-1][0])
    assert last_step < 0.3

    tracker = FruitTracker(TrackingConfig(match_iou=0.3, min_hits=3))
    ids = ids_over(tracker, frames)
    assert tracker.confirmed_count == 1
    assert all(frame == [1] for frame in ids[2:])


def test_new_track_cannot_associate_extreme_motion_before_velocity_is_known() -> None:
    # Documented limitation (shared with Kalman-based trackers such as ByteTrack):
    # a new track has no velocity estimate, so an object that already moves ~75% of
    # its width per frame when it first appears is never linked.
    tracker = FruitTracker(TrackingConfig(match_iou=0.3, min_hits=2))
    ids_over(tracker, [[moving_box(f, speed=30)] for f in range(10)])
    assert tracker.confirmed_count == 0


def test_short_occlusion_does_not_create_a_new_identity() -> None:
    tracker = FruitTracker(TrackingConfig(max_age=10))
    frames = [[moving_box(f)] for f in range(10)]
    frames += [[] for _ in range(5)]  # fruit hidden behind a leaf for 5 frames
    frames += [[moving_box(f)] for f in range(15, 25)]
    ids = ids_over(tracker, frames)
    assert ids[-1] == [1]
    assert tracker.confirmed_count == 1


def test_track_expires_after_max_age_and_reappearance_gets_new_id() -> None:
    tracker = FruitTracker(TrackingConfig(max_age=3, min_hits=2))
    frames = [[moving_box(0, speed=0)] for _ in range(5)]
    frames += [[] for _ in range(4)]  # longer than max_age
    frames += [[moving_box(0, speed=0)] for _ in range(3)]
    ids = ids_over(tracker, frames)
    assert ids[4] == [1]
    assert ids[-1] == [2]
    assert tracker.confirmed_count == 2


def test_single_frame_false_positive_is_never_confirmed() -> None:
    tracker = FruitTracker(TrackingConfig(min_hits=3))
    frames = [[moving_box(0, x0=200)]] + [[] for _ in range(5)]
    ids_over(tracker, frames)
    assert tracker.confirmed_count == 0
    assert tracker.tracks == []


def test_flickering_detection_must_be_consecutive_to_confirm() -> None:
    tracker = FruitTracker(TrackingConfig(min_hits=3))
    det = moving_box(0, speed=0)
    ids_over(tracker, [[det], [det], [], [det], [det], []])
    assert tracker.confirmed_count == 0


def test_two_separate_objects_get_two_identities() -> None:
    tracker = FruitTracker()
    frames = [[moving_box(f, y=10), moving_box(f, y=150, class_name="orange")] for f in range(10)]
    ids = ids_over(tracker, frames)
    assert ids[-1] == [1, 2]
    assert tracker.confirmed_count == 2


def test_identities_follow_objects_when_input_order_changes() -> None:
    tracker = FruitTracker()
    frames = []
    for f in range(10):
        a, b = moving_box(f, y=10), moving_box(f, y=150)
        frames.append([a, b] if f % 2 == 0 else [b, a])
    ids = ids_over(tracker, frames)
    assert ids[8] == [1, 2]
    assert ids[9] == [2, 1]


def test_low_confidence_detection_extends_but_never_starts_tracks() -> None:
    cfg = TrackingConfig(high_threshold=0.5, min_hits=2, max_age=1)
    tracker = FruitTracker(cfg)
    # Low-confidence-only object: never tracked.
    ids = ids_over(tracker, [[moving_box(f, conf=0.3)] for f in range(5)])
    assert all(frame == [None] for frame in ids)
    assert tracker.confirmed_count == 0

    # Confident object that becomes weak (partial occlusion): its track survives far
    # longer than max_age=1 because the weak detections keep matching it.
    tracker = FruitTracker(cfg)
    frames = [[moving_box(f, conf=0.9)] for f in range(3)]
    frames += [[moving_box(f, conf=0.3)] for f in range(3, 10)]
    ids = ids_over(tracker, frames)
    assert ids[-1] == [1]
    assert tracker.confirmed_count == 1


def test_removed_tracks_are_pruned() -> None:
    tracker = FruitTracker(TrackingConfig(max_age=2, min_hits=1))
    ids_over(tracker, [[moving_box(0)], [], [], []])
    assert tracker.tracks == []


def test_confirmed_track_state() -> None:
    tracker = FruitTracker(TrackingConfig(min_hits=2))
    ids_over(tracker, [[moving_box(0)], [moving_box(1)]])
    (track,) = tracker.tracks
    assert (track.track_id, track.hits, track.misses) == (1, 2, 0)


def test_output_preserves_input_order_and_length() -> None:
    tracker = FruitTracker()
    dets = [moving_box(0, y=10), moving_box(0, y=100, conf=0.2), moving_box(0, y=200)]
    out = tracker.update(dets)
    assert [t.detection for t in out] == dets


@pytest.mark.parametrize("speed", [0, 5, -5])
def test_prediction_is_stable_for_stationary_and_moving_objects(speed: float) -> None:
    tracker = FruitTracker(TrackingConfig(min_hits=1))
    ids_over(tracker, [[moving_box(f, x0=300, speed=speed)] for f in range(20)])
    (track,) = tracker.tracks
    expected = moving_box(20, x0=300, speed=speed).box
    track.misses += 1  # advance one frame without a detection
    assert iou_matrix(track.predicted_xyxy(), np.array(expected))[0, 0] > 0.9
