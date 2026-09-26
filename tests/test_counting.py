from conftest import make_detection
from fruit_counter.config import TrackingConfig
from fruit_counter.counting import UniqueCounter
from fruit_counter.tracker import FruitTracker, TrackedDetection


def tracked(track_id: int | None, class_name: str = "apple", conf: float = 0.9):
    return TrackedDetection(make_detection(0, 0, 10, 10, conf, class_name), track_id)


def test_same_fruit_over_100_frames_is_counted_once() -> None:
    counter = UniqueCounter()
    for frame in range(100):
        counter.update(frame, [tracked(1)])
    assert counter.total == 1
    (obj,) = counter.objects
    assert (obj.first_frame, obj.last_frame, obj.observations) == (0, 99, 100)


def test_unconfirmed_detections_are_not_counted() -> None:
    counter = UniqueCounter()
    counter.update(0, [tracked(None), tracked(None)])
    assert counter.total == 0
    assert counter.counts_by_class == {}


def test_update_reports_newly_counted_ids() -> None:
    counter = UniqueCounter()
    assert counter.update(0, [tracked(1), tracked(None)]) == [1]
    assert counter.update(1, [tracked(1), tracked(2)]) == [2]
    assert counter.update(2, [tracked(1), tracked(2)]) == []
    assert counter.total == 2


def test_class_is_decided_by_confidence_weighted_vote() -> None:
    counter = UniqueCounter()
    # Mislabelled as orange twice with low confidence, apple three times confidently.
    frames = [("orange", 0.3), ("apple", 0.8), ("orange", 0.35), ("apple", 0.9), ("apple", 0.7)]
    for i, (name, conf) in enumerate(frames):
        counter.update(i, [tracked(1, name, conf)])
    counter.update(5, [tracked(2, "banana")])

    assert counter.counts_by_class == {"apple": 1, "banana": 1}
    obj = counter.objects[0]
    assert obj.class_name == "apple"
    assert obj.mean_confidence == sum(c for _, c in frames) / len(frames)
    assert obj.to_dict()["class_name"] == "apple"


def test_class_vote_tie_is_deterministic() -> None:
    counter = UniqueCounter()
    counter.update(0, [tracked(1, "orange", 0.5)])
    counter.update(1, [tracked(1, "apple", 0.5)])
    assert counter.objects[0].class_name == "apple"


def test_detection_count_differs_from_unique_count_with_tracker() -> None:
    """End-to-end check of the core requirement using the real tracker.

    Three fruits cross the view at different times over 60 frames. Summing
    per-frame detections massively overcounts; unique counting does not.
    """
    tracker = FruitTracker(TrackingConfig(min_hits=3, max_age=5))
    counter = UniqueCounter()
    fruit_spans = {"apple": range(0, 30), "orange": range(20, 50), "banana": range(40, 60)}
    per_frame_sum = 0
    for frame in range(60):
        dets = [
            make_detection(5 * frame, 60 * k, 5 * frame + 40, 60 * k + 40, 0.85, name)
            for k, (name, span) in enumerate(fruit_spans.items())
            if frame in span
        ]
        per_frame_sum += len(dets)
        counter.update(frame, tracker.update(dets))

    assert per_frame_sum == 80
    assert counter.total == 3
    assert counter.counts_by_class == {"apple": 1, "banana": 1, "orange": 1}
