"""Engine module for Music Flamingo inference."""

from flamingo_inference.engine.async_flamingo import AsyncFlamingoEngine, quick_caption

__all__ = [
    "AsyncFlamingoEngine",
    "quick_caption",
]
