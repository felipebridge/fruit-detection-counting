import numpy as np
import pytest

from fruit_counter.structures import BoundingBox, Detection, iou_matrix


def test_box_geometry() -> None:
    box = BoundingBox(10, 20, 50, 100)
    assert box.width == 40
    assert box.height == 80
    assert box.area == 3200
    assert box.center == (30, 60)
    assert box.to_cxcywh() == (30, 60, 40, 80)


def test_from_cxcywh_round_trips() -> None:
    box = BoundingBox(10, 20, 50, 100)
    assert BoundingBox.from_cxcywh(*box.to_cxcywh()) == box


@pytest.mark.parametrize(
    "coords",
    [(10, 10, 5, 20), (10, 10, 20, 5), (0, 0, float("nan"), 1), (0, 0, float("inf"), 1)],
)
def test_invalid_boxes_are_rejected(coords: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        BoundingBox(*coords)


def test_iou_known_values() -> None:
    a = BoundingBox(0, 0, 10, 10)
    assert a.iou(a) == pytest.approx(1.0)
    assert a.iou(BoundingBox(20, 20, 30, 30)) == 0.0
    # Half overlap: intersection 50, union 150.
    assert a.iou(BoundingBox(5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_iou_matrix_shape_and_values() -> None:
    a = np.array([[0, 0, 10, 10], [100, 100, 110, 110]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [200, 200, 210, 210]])
    matrix = iou_matrix(a, b)
    assert matrix.shape == (2, 3)
    np.testing.assert_allclose(matrix[0], [1.0, 1 / 3, 0.0])
    np.testing.assert_allclose(matrix[1], [0.0, 0.0, 0.0])


def test_iou_matrix_handles_empty_inputs_and_degenerate_boxes() -> None:
    assert iou_matrix(np.empty((0, 4)), np.array([[0, 0, 1, 1]])).shape == (0, 1)
    assert iou_matrix(np.array([[0, 0, 1, 1]]), np.empty((0, 4))).shape == (1, 0)
    zero_area = np.array([[5, 5, 5, 5]])
    assert iou_matrix(zero_area, zero_area)[0, 0] == 0.0


def test_clip_keeps_box_inside_image() -> None:
    clipped = BoundingBox(-10, -5, 700, 500).clip(640, 480)
    assert clipped.to_xyxy() == (0, 0, 640, 480)
    outside = BoundingBox(800, 600, 900, 700).clip(640, 480)
    assert outside.area == 0


def test_detection_serialisation() -> None:
    det = Detection(BoundingBox(1.234, 2, 30.06, 40), 0.87654, 47, "apple")
    assert det.to_dict() == {
        "class_id": 47,
        "class_name": "apple",
        "confidence": 0.8765,
        "box_xyxy": [1.2, 2.0, 30.1, 40.0],
    }
