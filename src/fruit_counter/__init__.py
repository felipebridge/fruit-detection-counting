"""Fruit Detection Counting: detect and count fruits in images and videos.

Python API::

    from fruit_counter import FruitCountingRunner, load_config, resolve_source

    config = load_config("configs/default.yaml")
    report = FruitCountingRunner(config).run(resolve_source("orchard.mp4"))
    print(report.fruit_count)

For in-memory frames (e.g. behind a web service) use :class:`ImagePipeline` or
:class:`VideoPipeline` directly with any :class:`Detector`.
"""

__version__ = "0.1.0"

from fruit_counter.config import AppConfig, load_config
from fruit_counter.detector import Detector, YoloDetector
from fruit_counter.exceptions import FruitCounterError
from fruit_counter.pipeline import ImagePipeline, VideoPipeline
from fruit_counter.runner import FruitCountingRunner, RunReport
from fruit_counter.sources import resolve_source

__all__ = [
    "AppConfig",
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
