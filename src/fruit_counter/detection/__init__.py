"""Object detectors."""

from fruit_counter.detection.base import Detector
from fruit_counter.detection.yolo import YoloDetector

__all__ = ["Detector", "YoloDetector"]
