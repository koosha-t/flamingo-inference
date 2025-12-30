"""Priority queue for inference requests."""

from __future__ import annotations

import heapq
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flamingo_inference.executor.abstract import InferenceRequest

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PrioritizedRequest:
    """Wrapper for priority queue ordering.

    Higher priority values are processed first.
    For equal priorities, earlier creation times are processed first.
    """

    priority: int = field(compare=True)
    created_at: float = field(compare=True)
    request: "InferenceRequest" = field(compare=False)

    def __init__(self, request: "InferenceRequest"):
        # Negate priority for max-heap behavior (heapq is min-heap)
        self.priority = -request.priority
        self.created_at = request.created_at
        self.request = request


class RequestQueue:
    """Thread-safe priority queue for inference requests.

    Supports FIFO and priority-based scheduling.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize the queue.

        Args:
            max_size: Maximum number of pending requests
        """
        self.max_size = max_size
        self._heap: list[PrioritizedRequest] = []
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._request_ids: set[str] = set()

    def put(self, request: "InferenceRequest", block: bool = True, timeout: float | None = None) -> bool:
        """Add a request to the queue.

        Args:
            request: The request to add
            block: Whether to block if queue is full
            timeout: Maximum time to wait if blocking

        Returns:
            True if added, False if queue is full
        """
        with self._not_full:
            if len(self._heap) >= self.max_size:
                if not block:
                    return False
                if not self._not_full.wait(timeout=timeout):
                    return False

            # Check for duplicate
            if request.request_id in self._request_ids:
                logger.warning(f"Duplicate request ID: {request.request_id}")
                return False

            item = PrioritizedRequest(request)
            heapq.heappush(self._heap, item)
            self._request_ids.add(request.request_id)
            self._not_empty.notify()
            return True

    def get(self, block: bool = True, timeout: float | None = None) -> "InferenceRequest | None":
        """Get the highest priority request.

        Args:
            block: Whether to block if queue is empty
            timeout: Maximum time to wait if blocking

        Returns:
            The request, or None if queue is empty
        """
        with self._not_empty:
            if not self._heap:
                if not block:
                    return None
                if not self._not_empty.wait(timeout=timeout):
                    return None

            if not self._heap:
                return None

            item = heapq.heappop(self._heap)
            self._request_ids.discard(item.request.request_id)
            self._not_full.notify()
            return item.request

    def get_batch(
        self,
        max_batch_size: int,
        max_total_duration: float,
        timeout: float | None = None,
    ) -> list["InferenceRequest"]:
        """Get a batch of requests for processing.

        Groups requests while respecting batch size and total audio duration limits.

        Args:
            max_batch_size: Maximum number of requests in batch
            max_total_duration: Maximum total audio duration in seconds
            timeout: Time to wait for at least one request

        Returns:
            List of requests for the batch
        """
        with self._not_empty:
            # Wait for at least one request
            if not self._heap:
                if timeout is not None:
                    self._not_empty.wait(timeout=timeout)
                else:
                    self._not_empty.wait()

            if not self._heap:
                return []

            batch = []
            total_duration = 0.0

            # Collect requests for batch
            remaining = []

            while self._heap and len(batch) < max_batch_size:
                item = heapq.heappop(self._heap)
                request = item.request

                # Check if adding this request would exceed duration limit
                if total_duration + request.audio_duration > max_total_duration:
                    remaining.append(item)
                    continue

                batch.append(request)
                self._request_ids.discard(request.request_id)
                total_duration += request.audio_duration

            # Put back requests that didn't fit
            for item in remaining:
                heapq.heappush(self._heap, item)

            if batch:
                self._not_full.notify_all()

            return batch

    def peek(self) -> "InferenceRequest | None":
        """Peek at the highest priority request without removing it.

        Returns:
            The request, or None if queue is empty
        """
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0].request

    def remove(self, request_id: str) -> bool:
        """Remove a specific request from the queue.

        Args:
            request_id: ID of request to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if request_id not in self._request_ids:
                return False

            # Find and remove the request
            self._heap = [
                item for item in self._heap
                if item.request.request_id != request_id
            ]
            heapq.heapify(self._heap)
            self._request_ids.discard(request_id)
            self._not_full.notify()
            return True

    def clear(self) -> int:
        """Clear all requests from the queue.

        Returns:
            Number of requests cleared
        """
        with self._lock:
            count = len(self._heap)
            self._heap.clear()
            self._request_ids.clear()
            self._not_full.notify_all()
            return count

    def __len__(self) -> int:
        """Get the number of pending requests."""
        with self._lock:
            return len(self._heap)

    def __contains__(self, request_id: str) -> bool:
        """Check if a request is in the queue."""
        with self._lock:
            return request_id in self._request_ids

    @property
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        with self._lock:
            return len(self._heap) == 0

    @property
    def is_full(self) -> bool:
        """Check if the queue is at capacity."""
        with self._lock:
            return len(self._heap) >= self.max_size
