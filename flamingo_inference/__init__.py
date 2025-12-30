"""Music Flamingo Inference Server.

High-availability inference server for the Music Flamingo model with
OpenAI-compatible REST API, batch processing, and multi-GPU support.
"""

__version__ = "0.1.0"

from flamingo_inference.engine.async_flamingo import AsyncFlamingoEngine
from flamingo_inference.config.engine_config import FlamingoEngineConfig
from flamingo_inference.config.server_config import ServerConfig

__all__ = [
    "__version__",
    "AsyncFlamingoEngine",
    "FlamingoEngineConfig",
    "ServerConfig",
]
