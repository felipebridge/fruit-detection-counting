import numpy as np
import pytest

from conftest import ScriptedDetector, make_detection
from fruit_counter.counting import count_by_class
from fruit_counter.exceptions import InputError
from fruit_counter.pipeline import ImagePipeline
from fruit_counter.visualization import FRUIT_COLORS, PALETTE, TENTATIVE_COLOR, Annotator

APPLE = FRUIT_COLORS["apple"]


def near(pixel: np.ndarray, color: tuple[int, int, int], tolerance: int = 40) -> bool:
    """Anti-aliased edges are close to, not exactly, the drawing colour."""
    return bool(np.abs(pixel.astype(int) - np.array(color)).max() <= tolerance)


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


def test_grayscale_input_is_annotated_in_colour() -> None:
    gray = np.zeros((240, 320), dtype=np.uint8)
    pipeline = ImagePipeline(ScriptedDetector([[make_detection(180, 120, 300, 220)]]))
    annotated = pipeline.annotate(gray, pipeline.process(gray))
    assert annotated.shape == (240, 320, 3)
    assert near(annotated[170, 300], APPLE)


def test_bgr_input_is_annotated_in_place(blank_frame: np.ndarray) -> None:
    pipeline = ImagePipeline(ScriptedDetector([[make_detection(20, 20, 80, 80)]]))
    assert pipeline.annotate(blank_frame, pipeline.process(blank_frame)) is blank_frame


def test_invalid_input_is_rejected_before_inference() -> None:
    detector = ScriptedDetector([[]])
    with pytest.raises(InputError):
        ImagePipeline(detector).process(np.zeros((0, 0, 3), dtype=np.uint8))
    assert detector.calls == 0


def test_annotation_draws_boxes_and_summary(blank_frame: np.ndarray) -> None:
    det = make_detection(50, 50, 150, 150, 0.9, "apple")
    pipeline = ImagePipeline(ScriptedDetector([[det]]))
    result = pipeline.process(blank_frame)
    annotated = pipeline.annotate(blank_frame, result)

    assert annotated.shape == (240, 320, 3)
    # The right edge of the box is drawn in the fruit's colour.
    assert near(annotated[100, 150], APPLE)
    # The count card writes (anti-aliased) white text in the top-left corner.
    assert (annotated[:100, :200].min(axis=2) > 200).any()


def test_unconfirmed_tracks_are_drawn_grey(blank_frame: np.ndarray) -> None:
    dets = [make_detection(50, 50, 100, 100), make_detection(200, 50, 250, 100)]
    annotator = Annotator()
    annotated = annotator.draw(blank_frame, dets, track_ids=[7, None])
    assert near(annotated[75, 100], annotator.color_for(dets[0].class_name))
    assert near(annotated[75, 250], TENTATIVE_COLOR)


def test_track_ids_length_must_match(blank_frame: np.ndarray) -> None:
    with pytest.raises(ValueError, match="same length"):
        Annotator().draw(blank_frame, [make_detection(0, 0, 5, 5)], track_ids=[])


def test_class_colours_are_distinct_and_stable() -> None:
    annotator = Annotator()
    # COCO ids of apple and orange: class_id % len(PALETTE) would nearly collide.
    first, second = annotator.color_for(47), annotator.color_for(49)
    assert first != second
    assert annotator.color_for(47) == first  # stable across frames
    assert len({annotator.color_for(i) for i in range(len(PALETTE))}) == len(PALETTE)


def test_known_fruits_have_fixed_colours() -> None:
    annotator = Annotator()
    assert annotator.color_for("orange") == FRUIT_COLORS["orange"]
    assert annotator.color_for("Apple") == APPLE
    assert annotator.color_for("kiwi") == PALETTE[0]  # others: order of first appearance
