"""OpenAI-compatible API schemas and handlers."""

from flamingo_inference.entrypoints.openai.protocol import (
    AnalyzeRequest,
    AnalyzeResponse,
    AudioData,
    AudioUrl,
    BatchJobRequest,
    BatchJobResponse,
    CaptionRequest,
    CaptionResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
    ModelInfo,
    ModelListResponse,
)

__all__ = [
    # Caption
    "CaptionRequest",
    "CaptionResponse",
    # Embedding
    "EmbeddingRequest",
    "EmbeddingResponse",
    # Chat
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    # Analyze
    "AnalyzeRequest",
    "AnalyzeResponse",
    # Batch
    "BatchJobRequest",
    "BatchJobResponse",
    # Models
    "ModelInfo",
    "ModelListResponse",
    # Audio
    "AudioData",
    "AudioUrl",
    # Error
    "ErrorResponse",
]
