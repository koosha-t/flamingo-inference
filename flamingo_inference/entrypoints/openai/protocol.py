"""OpenAI-compatible API schemas for Music Flamingo."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ============================================================================
# Common Types
# ============================================================================


class AudioData(BaseModel):
    """Audio data in a request."""

    data: str = Field(..., description="Base64-encoded audio data")
    format: str | None = Field(None, description="Audio format (wav, mp3, flac, etc.)")


class AudioUrl(BaseModel):
    """Audio URL reference."""

    url: str = Field(..., description="URL to audio file")


# ============================================================================
# Caption Endpoint
# ============================================================================


class CaptionRequest(BaseModel):
    """Request for /v1/audio/captions endpoint."""

    model: str = Field(
        default="nvidia/music-flamingo-hf",
        description="Model to use for captioning",
    )
    audio: AudioData | AudioUrl = Field(..., description="Audio input")
    style: Literal["detailed", "brief", "technical"] = Field(
        default="detailed",
        description="Caption style",
    )
    prompt: str | None = Field(
        None,
        description="Custom prompt (overrides style)",
    )
    max_tokens: int = Field(
        default=512,
        ge=1,
        le=4096,
        description="Maximum tokens to generate",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p sampling",
    )
    stream: bool = Field(
        default=False,
        description="Stream response (not currently supported)",
    )


class CaptionUsage(BaseModel):
    """Usage statistics for caption request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    audio_duration_seconds: float = 0.0


class CaptionResponse(BaseModel):
    """Response from /v1/audio/captions endpoint."""

    id: str = Field(..., description="Request ID")
    object: str = "audio.caption"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "nvidia/music-flamingo-hf"
    caption: str = Field(..., description="Generated caption")
    finish_reason: str = "stop"
    usage: CaptionUsage = Field(default_factory=CaptionUsage)


# ============================================================================
# Embedding Endpoint
# ============================================================================


class EmbeddingRequest(BaseModel):
    """Request for /v1/audio/embeddings endpoint."""

    model: str = Field(
        default="nvidia/music-flamingo-hf",
        description="Model to use",
    )
    audio: AudioData | AudioUrl = Field(..., description="Audio input")
    chunk_duration: float | None = Field(
        None,
        description="Chunk duration for long audio (seconds)",
    )
    overlap: float = Field(
        default=5.0,
        ge=0.0,
        description="Overlap between chunks (seconds)",
    )
    encoding_format: Literal["float", "base64"] = Field(
        default="float",
        description="Format for embedding output",
    )
    dimensions: int | None = Field(
        None,
        description="Truncate embedding dimensions (not supported)",
    )


class EmbeddingData(BaseModel):
    """Single embedding in response."""

    object: str = "embedding"
    index: int = 0
    embedding: list[list[float]] | str = Field(
        ...,
        description="Embedding data (frames x dims) or base64",
    )
    frame_timestamps: list[float] | None = None


class EmbeddingUsage(BaseModel):
    """Usage statistics for embedding request."""

    prompt_tokens: int = 0
    total_tokens: int = 0
    audio_duration_seconds: float = 0.0
    num_frames: int = 0


class EmbeddingResponse(BaseModel):
    """Response from /v1/audio/embeddings endpoint."""

    id: str = Field(..., description="Request ID")
    object: str = "list"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "nvidia/music-flamingo-hf"
    data: list[EmbeddingData] = Field(..., description="Embedding data")
    usage: EmbeddingUsage = Field(default_factory=EmbeddingUsage)


# ============================================================================
# Chat Completions Endpoint (Audio Q&A)
# ============================================================================


class AudioContent(BaseModel):
    """Audio content in chat message."""

    type: Literal["audio"] = "audio"
    audio: AudioData | AudioUrl


class TextContent(BaseModel):
    """Text content in chat message."""

    type: Literal["text"] = "text"
    text: str


class ChatMessage(BaseModel):
    """Chat message with optional audio."""

    role: Literal["system", "user", "assistant"] = Field(..., description="Message role")
    content: str | list[TextContent | AudioContent] = Field(
        ...,
        description="Message content",
    )


class ChatCompletionRequest(BaseModel):
    """Request for /v1/chat/completions endpoint."""

    model: str = Field(
        default="nvidia/music-flamingo-hf",
        description="Model to use",
    )
    messages: list[ChatMessage] = Field(..., description="Chat messages")
    max_tokens: int = Field(default=2048, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop: list[str] | None = None
    stream: bool = False
    n: int = Field(default=1, ge=1, le=1)


class ChatChoice(BaseModel):
    """Single chat completion choice."""

    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"
    logprobs: Any | None = None


class ChatUsage(BaseModel):
    """Usage statistics for chat completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    audio_duration_seconds: float = 0.0


class ChatCompletionResponse(BaseModel):
    """Response from /v1/chat/completions endpoint."""

    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "nvidia/music-flamingo-hf"
    choices: list[ChatChoice]
    usage: ChatUsage = Field(default_factory=ChatUsage)


# ============================================================================
# Analyze Endpoint (Caption + Embedding)
# ============================================================================


class AnalyzeRequest(BaseModel):
    """Request for /v1/audio/analyze endpoint."""

    model: str = Field(
        default="nvidia/music-flamingo-hf",
        description="Model to use",
    )
    audio: AudioData | AudioUrl = Field(..., description="Audio input")
    prompt: str | None = Field(
        None,
        description="Custom caption prompt",
    )
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    include_embeddings: bool = Field(
        default=True,
        description="Include audio embeddings in response",
    )
    embedding_format: Literal["float", "base64"] = "float"


class AnalyzeResponse(BaseModel):
    """Response from /v1/audio/analyze endpoint."""

    id: str
    object: str = "audio.analysis"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "nvidia/music-flamingo-hf"
    caption: str
    finish_reason: str = "stop"
    embedding: EmbeddingData | None = None
    usage: CaptionUsage = Field(default_factory=CaptionUsage)


# ============================================================================
# Batch Job Endpoints
# ============================================================================


class BatchJobStatus(str, Enum):
    """Status of a batch job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchJobRequest(BaseModel):
    """Request for /v1/batch/jobs endpoint."""

    model: str = Field(
        default="nvidia/music-flamingo-hf",
        description="Model to use",
    )
    input_manifest: str = Field(
        ...,
        description="Path or URL to input manifest (JSONL)",
    )
    output_dir: str = Field(
        ...,
        description="Directory for output files",
    )
    task_type: Literal["caption", "embed", "analyze"] = Field(
        default="analyze",
        description="Type of task to run",
    )
    caption_style: str = "detailed"
    max_tokens: int = 512
    temperature: float = 0.7


class BatchJobProgress(BaseModel):
    """Progress information for batch job."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    pending: int = 0


class BatchJobResponse(BaseModel):
    """Response for batch job status."""

    id: str
    object: str = "batch.job"
    created: int = Field(default_factory=lambda: int(time.time()))
    status: BatchJobStatus = BatchJobStatus.PENDING
    model: str = "nvidia/music-flamingo-hf"
    task_type: str = "analyze"
    progress: BatchJobProgress = Field(default_factory=BatchJobProgress)
    output_dir: str | None = None
    error: str | None = None
    started_at: int | None = None
    completed_at: int | None = None


# ============================================================================
# Model Info Endpoint
# ============================================================================


class ModelInfo(BaseModel):
    """Model information."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "nvidia"


class ModelListResponse(BaseModel):
    """Response for /v1/models endpoint."""

    object: str = "list"
    data: list[ModelInfo]


# ============================================================================
# Error Response
# ============================================================================


class ErrorDetail(BaseModel):
    """Error detail."""

    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    """Error response."""

    error: ErrorDetail
