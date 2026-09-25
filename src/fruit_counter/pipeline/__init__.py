"""In-memory processing pipelines (no file I/O)."""

from fruit_counter.pipeline.image import ImagePipeline, ImageResult
from fruit_counter.pipeline.video import FrameResult, VideoPipeline, VideoSummary

__all__ = ["FrameResult", "ImagePipeline", "ImageResult", "VideoPipeline", "VideoSummary"]
