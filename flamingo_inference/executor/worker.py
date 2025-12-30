"""GPU worker for model execution."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from flamingo_inference.config import FlamingoEngineConfig

from flamingo_inference.executor.abstract import (
    InferenceRequest,
    InferenceResult,
    RequestType,
)
from flamingo_inference.models import FlamingoModel

logger = logging.getLogger(__name__)


@dataclass
class WorkerStats:
    """Statistics for a worker."""

    worker_id: int
    gpu_id: int
    requests_processed: int = 0
    total_processing_time: float = 0.0
    total_audio_duration: float = 0.0
    errors: int = 0
    last_request_time: float | None = None
    gpu_memory_allocated: int = 0
    gpu_memory_reserved: int = 0


class FlamingoWorker:
    """A single GPU worker that executes inference requests.

    Each worker loads a full copy of the model and handles
    requests assigned to it.
    """

    def __init__(
        self,
        worker_id: int,
        gpu_id: int,
        config: FlamingoEngineConfig,
    ):
        """Initialize the worker.

        Args:
            worker_id: Unique identifier for this worker
            gpu_id: CUDA device ID to use
            config: Engine configuration
        """
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.config = config

        self._model: FlamingoModel | None = None
        self._device: torch.device | None = None
        self._is_ready = False

        # Statistics
        self._stats = WorkerStats(worker_id=worker_id, gpu_id=gpu_id)

    def start(self) -> None:
        """Initialize the worker and load the model."""
        logger.info(f"Worker {self.worker_id}: Starting on GPU {self.gpu_id}")

        # Set device
        self._device = torch.device(f"cuda:{self.gpu_id}")

        # Load model
        logger.info(f"Worker {self.worker_id}: Loading model {self.config.model.name}")
        self._model = FlamingoModel.from_pretrained(
            self.config.model,
            device=self._device,
        )

        self._is_ready = True
        self._update_memory_stats()
        logger.info(
            f"Worker {self.worker_id}: Ready. "
            f"Memory: {self._stats.gpu_memory_allocated / 1e9:.2f}GB allocated"
        )

    def stop(self) -> None:
        """Shutdown the worker and free resources."""
        logger.info(f"Worker {self.worker_id}: Stopping")
        self._is_ready = False

        if self._model is not None:
            del self._model
            self._model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def execute(self, request: InferenceRequest) -> InferenceResult:
        """Execute a single inference request.

        Args:
            request: The inference request

        Returns:
            The inference result
        """
        if not self._is_ready or self._model is None:
            return InferenceResult(
                request_id=request.request_id,
                request_type=request.request_type,
                error="Worker not ready",
                success=False,
            )

        start_time = time.time()

        try:
            result = self._execute_request(request)
            result.processing_time = time.time() - start_time
            result.worker_id = self.worker_id

            # Update stats
            self._stats.requests_processed += 1
            self._stats.total_processing_time += result.processing_time
            self._stats.total_audio_duration += request.audio_duration
            self._stats.last_request_time = time.time()

        except Exception as e:
            logger.exception(f"Worker {self.worker_id}: Error executing request")
            self._stats.errors += 1
            result = InferenceResult(
                request_id=request.request_id,
                request_type=request.request_type,
                error=str(e),
                success=False,
                worker_id=self.worker_id,
                processing_time=time.time() - start_time,
            )

        self._update_memory_stats()
        return result

    def _execute_request(self, request: InferenceRequest) -> InferenceResult:
        """Internal request execution logic."""
        if request.audio is None:
            raise ValueError("Audio is required for inference")

        # Move audio to device
        audio = request.audio.to(self._device)

        result = InferenceResult(
            request_id=request.request_id,
            request_type=request.request_type,
            audio_duration=request.audio_duration,
        )

        # Execute based on request type
        if request.request_type == RequestType.GENERATE:
            gen_output = self._model.generate(
                audio=audio,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=request.do_sample,
                repetition_penalty=request.repetition_penalty,
            )
            result.text = gen_output.text
            result.tokens = gen_output.tokens
            result.logprobs = gen_output.logprobs
            result.finish_reason = gen_output.finish_reason

        elif request.request_type == RequestType.EMBED:
            embed_output = self._model.extract_embedding(
                audio=audio,
                sample_rate=self.config.audio.sample_rate,
            )
            result.embedding = embed_output.embedding
            result.frame_timestamps = embed_output.frame_timestamps

        elif request.request_type == RequestType.ANALYZE:
            # Both generation and embedding
            gen_output = self._model.generate(
                audio=audio,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=request.do_sample,
                repetition_penalty=request.repetition_penalty,
            )
            embed_output = self._model.extract_embedding(
                audio=audio,
                sample_rate=self.config.audio.sample_rate,
            )

            result.text = gen_output.text
            result.tokens = gen_output.tokens
            result.logprobs = gen_output.logprobs
            result.finish_reason = gen_output.finish_reason
            result.embedding = embed_output.embedding
            result.frame_timestamps = embed_output.frame_timestamps

        return result

    def execute_batch(
        self, requests: list[InferenceRequest]
    ) -> list[InferenceResult]:
        """Execute a batch of requests.

        Currently processes sequentially. Future versions could
        implement true batching for improved throughput.

        Args:
            requests: List of requests to execute

        Returns:
            List of results in the same order
        """
        results = []
        for request in requests:
            result = self.execute(request)
            results.append(result)
        return results

    def _update_memory_stats(self) -> None:
        """Update GPU memory statistics."""
        if torch.cuda.is_available() and self._model is not None:
            memory = self._model.get_memory_usage()
            self._stats.gpu_memory_allocated = memory["allocated"]
            self._stats.gpu_memory_reserved = memory["reserved"]

    @property
    def is_ready(self) -> bool:
        """Check if the worker is ready."""
        return self._is_ready

    @property
    def stats(self) -> WorkerStats:
        """Get worker statistics."""
        return self._stats

    def get_stats_dict(self) -> dict:
        """Get worker statistics as a dictionary."""
        return {
            "worker_id": self._stats.worker_id,
            "gpu_id": self._stats.gpu_id,
            "requests_processed": self._stats.requests_processed,
            "total_processing_time": self._stats.total_processing_time,
            "total_audio_duration": self._stats.total_audio_duration,
            "errors": self._stats.errors,
            "last_request_time": self._stats.last_request_time,
            "gpu_memory_allocated_gb": self._stats.gpu_memory_allocated / 1e9,
            "gpu_memory_reserved_gb": self._stats.gpu_memory_reserved / 1e9,
            "is_ready": self._is_ready,
            "avg_processing_time": (
                self._stats.total_processing_time / self._stats.requests_processed
                if self._stats.requests_processed > 0
                else 0.0
            ),
        }
