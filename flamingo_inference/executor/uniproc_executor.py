"""Single-process, single-GPU executor."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

from flamingo_inference.config import FlamingoEngineConfig
from flamingo_inference.executor.abstract import (
    Executor,
    InferenceRequest,
    InferenceResult,
)
from flamingo_inference.executor.worker import FlamingoWorker

logger = logging.getLogger(__name__)


class UniprocExecutor(Executor):
    """Single-process executor for single GPU deployment.

    This executor runs in the same process as the engine and
    is suitable for development and single-GPU production deployments.
    """

    def __init__(self, config: FlamingoEngineConfig):
        """Initialize the executor.

        Args:
            config: Engine configuration
        """
        super().__init__(config)

        self._worker: FlamingoWorker | None = None
        self._gpu_id: int = 0

        # Determine GPU ID
        if config.executor.gpu_ids:
            self._gpu_id = config.executor.gpu_ids[0]
        elif torch.cuda.is_available():
            self._gpu_id = 0
        else:
            raise RuntimeError("No CUDA device available")

    def start(self) -> None:
        """Start the executor and initialize the worker."""
        if self._is_running:
            logger.warning("Executor already running")
            return

        logger.info(f"UniprocExecutor: Starting with GPU {self._gpu_id}")

        self._worker = FlamingoWorker(
            worker_id=0,
            gpu_id=self._gpu_id,
            config=self.config,
        )
        self._worker.start()
        self._is_running = True

        logger.info("UniprocExecutor: Ready")

    def stop(self) -> None:
        """Stop the executor and shutdown the worker."""
        if not self._is_running:
            return

        logger.info("UniprocExecutor: Stopping")

        if self._worker is not None:
            self._worker.stop()
            self._worker = None

        self._is_running = False
        logger.info("UniprocExecutor: Stopped")

    def execute(self, request: InferenceRequest) -> InferenceResult:
        """Execute a single inference request.

        Args:
            request: The inference request

        Returns:
            The inference result
        """
        if not self._is_running or self._worker is None:
            return InferenceResult(
                request_id=request.request_id,
                request_type=request.request_type,
                error="Executor not running",
                success=False,
            )

        return self._worker.execute(request)

    def execute_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        """Execute a batch of inference requests.

        Args:
            requests: List of inference requests

        Returns:
            List of inference results
        """
        if not self._is_running or self._worker is None:
            return [
                InferenceResult(
                    request_id=req.request_id,
                    request_type=req.request_type,
                    error="Executor not running",
                    success=False,
                )
                for req in requests
            ]

        return self._worker.execute_batch(requests)

    @property
    def num_workers(self) -> int:
        """Get the number of active workers."""
        return 1 if self._is_running and self._worker is not None else 0

    @property
    def is_ready(self) -> bool:
        """Check if the executor is ready."""
        return (
            self._is_running
            and self._worker is not None
            and self._worker.is_ready
        )

    def get_worker_stats(self) -> list[dict]:
        """Get statistics for each worker."""
        if self._worker is None:
            return []
        return [self._worker.get_stats_dict()]
