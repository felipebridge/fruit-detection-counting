"""Fruit detection and counting in images and videos."""

__version__ = "0.1.0"

from fruit_counter.config import AppConfig, load_config
from fruit_counter.detector import Detection, Detector, YoloDetector
from fruit_counter.exceptions import FruitCounterError
from fruit_counter.pipeline import ImagePipeline, VideoPipeline
from fruit_counter.runner import FruitCountingRunner, RunReport
from fruit_counter.sources import resolve_source

__all__ = [
    "AppConfig",
    "Detection",
    "Detector",
    "FruitCounterError",
    "FruitCountingRunner",
    "ImagePipeline",
    "RunReport",
    "VideoPipeline",
    "YoloDetector",
    "__version__",
    "load_config",
    "resolve_source",
]
