"""Request scheduler for Music Flamingo inference."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from flamingo_inference.executor.abstract import InferenceResult

from flamingo_inference.config import SchedulerConfig, SchedulerPolicy
from flamingo_inference.executor.abstract import (
    InferenceRequest,
    RequestStatus,
)
from flamingo_inference.scheduler.request_queue import RequestQueue

logger = logging.getLogger(__name__)


@dataclass
class SchedulerStats:
    """Statistics for the scheduler."""

    requests_submitted: int = 0
    requests_completed: int = 0
    requests_failed: int = 0
    requests_aborted: int = 0
    total_wait_time: float = 0.0
    total_processing_time: float = 0.0
    total_audio_duration: float = 0.0
    current_queue_size: int = 0
    current_running: int = 0


@dataclass
class PendingResult:
    """Tracks a pending request and its completion future."""

    request: InferenceRequest
    future: asyncio.Future
    submitted_at: float = field(default_factory=time.time)


class FlamingoScheduler:
    """Scheduler for managing inference request lifecycle.

    Handles:
    - Request submission and queuing
    - Priority-based scheduling
    - Audio-aware batching
    - Request lifecycle tracking
    """

    def __init__(
        self,
        config: SchedulerConfig,
        executor_callback: Callable[[list[InferenceRequest]], list["InferenceResult"]],
    ):
        """Initialize the scheduler.

        Args:
            config: Scheduler configuration
            executor_callback: Callback to execute batches of requests
        """
        self.config = config
        self._executor_callback = executor_callback

        # Request queue
        self._queue = RequestQueue(max_size=config.max_waiting_requests)

        # Tracking
        self._pending: dict[str, PendingResult] = {}
        self._running: dict[str, InferenceRequest] = {}
        self._stats = SchedulerStats()

        # Threading
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._scheduler_thread: threading.Thread | None = None

        # Async support
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start the scheduler.

        Args:
            loop: Event loop for async operations
        """
        if self._scheduler_thread is not None:
            logger.warning("Scheduler already running")
            return

        self._loop = loop or asyncio.get_event_loop()
        self._shutdown.clear()

        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="FlamingoScheduler",
            daemon=True,
        )
        self._scheduler_thread.start()
        logger.info("Scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._scheduler_thread is None:
            return

        logger.info("Stopping scheduler...")
        self._shutdown.set()

        # Abort pending requests
        with self._lock:
            for pending in self._pending.values():
                if not pending.future.done():
                    pending.future.cancel()
            self._pending.clear()

        self._scheduler_thread.join(timeout=5.0)
        self._scheduler_thread = None
        logger.info("Scheduler stopped")

    async def submit(self, request: InferenceRequest) -> "InferenceResult":
        """Submit a request and wait for completion.

        Args:
            request: The inference request

        Returns:
            The inference result

        Raises:
            asyncio.CancelledError: If the request was cancelled
            Exception: If an error occurred during processing
        """
        # Create future for result
        future = self._loop.create_future()

        # Track pending request
        pending = PendingResult(request=request, future=future)
        with self._lock:
            self._pending[request.request_id] = pending
            self._stats.requests_submitted += 1

        # Add to queue
        request.status = RequestStatus.WAITING
        if not self._queue.put(request, block=False):
            with self._lock:
                del self._pending[request.request_id]
            raise RuntimeError("Request queue is full")

        self._stats.current_queue_size = len(self._queue)

        # Wait for result
        try:
            result = await future
            return result
        except asyncio.CancelledError:
            self.abort(request.request_id)
            raise

    def submit_sync(self, request: InferenceRequest) -> "InferenceResult":
        """Submit a request synchronously (blocking).

        Args:
            request: The inference request

        Returns:
            The inference result
        """
        # Create event for completion
        done_event = threading.Event()
        result_holder: list = []

        def on_complete(result):
            result_holder.append(result)
            done_event.set()

        # Track pending request
        request.status = RequestStatus.WAITING
        self._pending[request.request_id] = PendingResult(
            request=request,
            future=None,  # type: ignore
        )

        with self._lock:
            self._stats.requests_submitted += 1

        # Add to queue
        if not self._queue.put(request, block=True, timeout=10.0):
            with self._lock:
                del self._pending[request.request_id]
            raise RuntimeError("Request queue is full")

        self._stats.current_queue_size = len(self._queue)

        # Wait for completion (polling)
        while not self._shutdown.is_set():
            if request.request_id not in self._pending:
                break
            time.sleep(0.01)

        if result_holder:
            return result_holder[0]

        raise RuntimeError("Request processing failed")

    def abort(self, request_id: str) -> bool:
        """Abort a pending request.

        Args:
            request_id: ID of request to abort

        Returns:
            True if request was aborted
        """
        with self._lock:
            # Remove from queue if waiting
            if self._queue.remove(request_id):
                if request_id in self._pending:
                    pending = self._pending.pop(request_id)
                    pending.request.status = RequestStatus.ABORTED
                    self._stats.requests_aborted += 1
                    return True

            # Can't abort running requests
            if request_id in self._running:
                logger.warning(f"Cannot abort running request: {request_id}")
                return False

        return False

    def _scheduler_loop(self) -> None:
        """Main scheduler loop running in background thread."""
        logger.info("Scheduler loop started")

        while not self._shutdown.is_set():
            try:
                # Get batch of requests
                batch = self._queue.get_batch(
                    max_batch_size=self.config.max_batch_size,
                    max_total_duration=self.config.max_total_audio_duration,
                    timeout=self.config.batch_wait_timeout,
                )

                if not batch:
                    continue

                # Update status
                for request in batch:
                    request.status = RequestStatus.RUNNING
                    request.started_at = time.time()
                    with self._lock:
                        self._running[request.request_id] = request
                        self._stats.current_running = len(self._running)

                # Execute batch
                try:
                    results = self._executor_callback(batch)
                except Exception as e:
                    logger.exception("Executor error")
                    results = []
                    for request in batch:
                        from flamingo_inference.executor.abstract import InferenceResult
                        results.append(
                            InferenceResult(
                                request_id=request.request_id,
                                request_type=request.request_type,
                                error=str(e),
                                success=False,
                            )
                        )

                # Complete requests
                for result in results:
                    self._complete_request(result)

            except Exception:
                logger.exception("Scheduler loop error")
                time.sleep(0.1)

        logger.info("Scheduler loop stopped")

    def _complete_request(self, result: "InferenceResult") -> None:
        """Complete a request with its result."""
        request_id = result.request_id

        with self._lock:
            # Remove from running
            request = self._running.pop(request_id, None)
            self._stats.current_running = len(self._running)

            # Get pending entry
            pending = self._pending.pop(request_id, None)

            if request is not None:
                request.status = RequestStatus.FINISHED if result.success else RequestStatus.ERROR
                request.finished_at = time.time()

                # Update stats
                if result.success:
                    self._stats.requests_completed += 1
                else:
                    self._stats.requests_failed += 1

                if request.started_at:
                    wait_time = request.started_at - request.created_at
                    self._stats.total_wait_time += wait_time

                self._stats.total_processing_time += result.processing_time
                self._stats.total_audio_duration += result.audio_duration

        # Resolve future
        if pending is not None and pending.future is not None:
            if not pending.future.done():
                if self._loop is not None:
                    self._loop.call_soon_threadsafe(
                        pending.future.set_result, result
                    )

        self._stats.current_queue_size = len(self._queue)

    def get_stats(self) -> dict:
        """Get scheduler statistics."""
        with self._lock:
            return {
                "requests_submitted": self._stats.requests_submitted,
                "requests_completed": self._stats.requests_completed,
                "requests_failed": self._stats.requests_failed,
                "requests_aborted": self._stats.requests_aborted,
                "total_wait_time": self._stats.total_wait_time,
                "total_processing_time": self._stats.total_processing_time,
                "total_audio_duration": self._stats.total_audio_duration,
                "current_queue_size": self._stats.current_queue_size,
                "current_running": self._stats.current_running,
                "avg_wait_time": (
                    self._stats.total_wait_time / self._stats.requests_completed
                    if self._stats.requests_completed > 0
                    else 0.0
                ),
                "avg_processing_time": (
                    self._stats.total_processing_time / self._stats.requests_completed
                    if self._stats.requests_completed > 0
                    else 0.0
                ),
            }

    def get_request_status(self, request_id: str) -> RequestStatus | None:
        """Get the status of a request.

        Args:
            request_id: The request ID

        Returns:
            The request status, or None if not found
        """
        with self._lock:
            if request_id in self._queue:
                return RequestStatus.WAITING
            if request_id in self._running:
                return RequestStatus.RUNNING
            if request_id in self._pending:
                return self._pending[request_id].request.status
        return None
