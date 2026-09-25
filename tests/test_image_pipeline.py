import numpy as np
import pytest

from conftest import ScriptedDetector, make_detection
from fruit_counter.config import VisualizationConfig
from fruit_counter.counting import count_by_class
from fruit_counter.exceptions import InputError
from fruit_counter.pipeline import ImagePipeline
from fruit_counter.visualization import Annotator, color_for_class


def test_count_by_class_groups_and_sorts() -> None:
    detections = [
        make_detection(0, 0, 10, 10, class_name="orange"),
        make_detection(20, 0, 30, 10, class_name="apple"),
        make_detection(40, 0, 50, 10, class_name="orange"),
    ]
    assert count_by_class(detections) == {"apple": 1, "orange": 2}
    assert count_by_class([]) == {}


def test_image_result_counts_every_detection(blank_frame: np.ndarray) -> None:
    detections = [
        make_detection(10, 10, 60, 60, 0.9, "apple"),
        make_detection(100, 10, 150, 60, 0.8, "apple"),
        make_detection(200, 10, 250, 60, 0.4, "banana"),
    ]
    result = ImagePipeline(ScriptedDetector([detections])).process(blank_frame)

    assert result.fruit_count == 3
    assert result.counts_by_class == {"apple": 2, "banana": 1}
    assert (result.width, result.height) == (320, 240)
    assert result.inference_ms >= 0

    data = result.to_dict()
    assert data["fruit_count"] == 3
    assert len(data["detections"]) == 3
    assert data["detections"][2]["class_name"] == "banana"


def test_image_without_fruit_yields_zero(blank_frame: np.ndarray) -> None:
    result = ImagePipeline(ScriptedDetector([[]])).process(blank_frame)
    assert result.fruit_count == 0
    assert result.counts_by_class == {}


def test_grayscale_input_is_accepted() -> None:
    detector = ScriptedDetector([[]])
    ImagePipeline(detector).process(np.zeros((50, 60), dtype=np.uint8))
    assert detector.calls == 1


def test_invalid_input_is_rejected_before_inference() -> None:
    detector = ScriptedDetector([[]])
    with pytest.raises(InputError):
        ImagePipeline(detector).process(np.zeros((0, 0, 3), dtype=np.uint8))
    assert detector.calls == 0


def test_annotation_draws_boxes_without_mutating_input(blank_frame: np.ndarray) -> None:
    det = make_detection(50, 50, 150, 150, 0.9, "apple")
    pipeline = ImagePipeline(ScriptedDetector([[det]]))
    result = pipeline.process(blank_frame)
    annotated = pipeline.annotate(blank_frame, result)

    assert annotated.shape == blank_frame.shape
    assert not blank_frame.any(), "input frame must not be modified"
    # The right edge of the box is drawn in the class colour.
    assert tuple(annotated[100, 150]) == color_for_class(det.class_id)
    # The summary panel writes (anti-aliased) white text in the top-left corner.
    assert (annotated[:40, :200].min(axis=2) > 200).any()


def test_unconfirmed_tracks_are_drawn_grey(blank_frame: np.ndarray) -> None:
    dets = [make_detection(50, 50, 100, 100), make_detection(200, 50, 250, 100)]
    annotated = Annotator(VisualizationConfig(show_summary=False)).draw(
        blank_frame, dets, track_ids=[7, None]
    )
    assert tuple(annotated[75, 100]) == color_for_class(dets[0].class_id)
    assert tuple(annotated[75, 250]) == (160, 160, 160)


def test_track_ids_length_must_match(blank_frame: np.ndarray) -> None:
    with pytest.raises(ValueError, match="same length"):
        Annotator().draw(blank_frame, [make_detection(0, 0, 5, 5)], track_ids=[])


def test_label_respects_visualization_config() -> None:
    det = make_detection(0, 0, 5, 5, 0.876, "orange")
    full = Annotator(VisualizationConfig())
    minimal = Annotator(VisualizationConfig(show_confidence=False, show_track_ids=False))
    assert full._label(det, 3) == "#3 orange 0.88"
    assert minimal._label(det, 3) == "orange"
