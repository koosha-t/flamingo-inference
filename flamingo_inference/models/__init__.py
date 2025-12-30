"""Model wrappers for Music Flamingo inference."""

from flamingo_inference.models.flamingo_model import (
    EmbeddingOutput,
    FlamingoModel,
    GenerationOutput,
)

__all__ = [
    "FlamingoModel",
    "GenerationOutput",
    "EmbeddingOutput",
]
