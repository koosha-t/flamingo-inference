"""Abstract base class for executors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING
from uuid import uuid4

import torch

if TYPE_CHECKING:
    from flamingo_inference.config import FlamingoEngineConfig


class RequestType(str, Enum):
    """Type of inference request."""

    GENERATE = "generate"
    EMBED = "embed"
    ANALYZE = "analyze"  # Both generate and embed


class RequestStatus(str, Enum):
    """Status of a request in the processing pipeline."""

    PENDING = "pending"
    WAITING = "waiting"
    PREPROCESSING = "preprocessing"
    RUNNING = "running"
    POSTPROCESSING = "postprocessing"
    FINISHED = "finished"
    ERROR = "error"
    ABORTED = "aborted"


@dataclass
class InferenceRequest:
    """A single inference request."""

    request_id: str = field(default_factory=lambda: str(uuid4()))
    request_type: RequestType = RequestType.GENERATE

    # Audio input
    audio: torch.Tensor | None = None
    audio_duration: float = 0.0

    # Generation parameters
    prompt: str = ""
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.0

    # Embedding parameters
    return_embeddings: bool = False

    # Request metadata
    priority: int = 0  # Higher = more priority
    created_at: float = 0.0
    status: RequestStatus = RequestStatus.PENDING

    # Processing state
    worker_id: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None

    def __post_init__(self):
        if self.created_at == 0.0:
            import time
            self.created_at = time.time()


@dataclass
class InferenceResult:
    """Result of an inference request."""

    request_id: str
    request_type: RequestType

    # Generation output
    text: str | None = None
    tokens: list[int] | None = None
    logprobs: list[float] | None = None
    finish_reason: str | None = None

    # Embedding output
    embedding: torch.Tensor | None = None
    frame_timestamps: torch.Tensor | None = None

    # Metadata
    audio_duration: float = 0.0
    processing_time: float = 0.0
    worker_id: int | None = None

    # Error info
    error: str | None = None
    success: bool = True


class Executor(ABC):
    """Abstract base class for model executors.

    An executor manages one or more GPU workers and handles
    request execution.
    """

    def __init__(self, config: FlamingoEngineConfig):
        """Initialize the executor.

        Args:
            config: Engine configuration
        """
        self.config = config
        self._is_running = False

    @abstractmethod
    def start(self) -> None:
        """Start the executor and workers."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the executor and workers."""
        pass

    @abstractmethod
    def execute(self, request: InferenceRequest) -> InferenceResult:
        """Execute a single inference request.

        Args:
            request: The inference request to execute

        Returns:
            The inference result
        """
        pass

    @abstractmethod
    def execute_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        """Execute a batch of inference requests.

        Args:
            requests: List of inference requests

        Returns:
            List of inference results in the same order
        """
        pass

    @property
    @abstractmethod
    def num_workers(self) -> int:
        """Get the number of active workers."""
        pass

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the executor is ready to accept requests."""
        pass

    @abstractmethod
    def get_worker_stats(self) -> list[dict]:
        """Get statistics for each worker."""
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
