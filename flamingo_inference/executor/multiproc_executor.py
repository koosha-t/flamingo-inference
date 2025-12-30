"""Multi-process, multi-GPU executor with data parallelism."""

from __future__ import annotations

import logging
import multiprocessing as mp
from multiprocessing import Queue
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

from flamingo_inference.config import FlamingoEngineConfig
from flamingo_inference.executor.abstract import (
    Executor,
    InferenceRequest,
    InferenceResult,
    RequestStatus,
)

logger = logging.getLogger(__name__)


def _worker_process(
    worker_id: int,
    gpu_id: int,
    config_dict: dict,
    request_queue: Queue,
    result_queue: Queue,
    ready_event: mp.Event,
    shutdown_event: mp.Event,
):
    """Worker process function.

    Args:
        worker_id: Worker identifier
        gpu_id: CUDA device to use
        config_dict: Serialized configuration
        request_queue: Queue for incoming requests
        result_queue: Queue for outgoing results
        ready_event: Event to signal worker is ready
        shutdown_event: Event to signal shutdown
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from flamingo_inference.config import FlamingoEngineConfig
    from flamingo_inference.executor.worker import FlamingoWorker

    # Reconstruct config
    config = FlamingoEngineConfig._from_dict(config_dict)

    # Create and start worker
    worker = FlamingoWorker(
        worker_id=worker_id,
        gpu_id=0,  # Always 0 because we set CUDA_VISIBLE_DEVICES
        config=config,
    )

    try:
        worker.start()
        ready_event.set()

        logger.info(f"Worker {worker_id} (GPU {gpu_id}): Ready and waiting for requests")

        while not shutdown_event.is_set():
            try:
                # Wait for request with timeout
                try:
                    request_data = request_queue.get(timeout=0.1)
                except Exception:
                    continue

                if request_data is None:
                    # Shutdown signal
                    break

                # Deserialize request
                request = _deserialize_request(request_data)

                # Execute
                result = worker.execute(request)

                # Serialize and send result
                result_data = _serialize_result(result)
                result_queue.put(result_data)

            except Exception as e:
                logger.exception(f"Worker {worker_id}: Error processing request")

    finally:
        worker.stop()
        logger.info(f"Worker {worker_id}: Shutdown complete")


def _serialize_request(request: InferenceRequest) -> dict:
    """Serialize request for IPC."""
    return {
        "request_id": request.request_id,
        "request_type": request.request_type.value,
        "audio": request.audio.cpu().numpy().tobytes() if request.audio is not None else None,
        "audio_shape": list(request.audio.shape) if request.audio is not None else None,
        "audio_duration": request.audio_duration,
        "prompt": request.prompt,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "do_sample": request.do_sample,
        "repetition_penalty": request.repetition_penalty,
        "return_embeddings": request.return_embeddings,
        "priority": request.priority,
        "created_at": request.created_at,
    }


def _deserialize_request(data: dict) -> InferenceRequest:
    """Deserialize request from IPC."""
    import numpy as np
    from flamingo_inference.executor.abstract import RequestType

    audio = None
    if data["audio"] is not None:
        audio_np = np.frombuffer(data["audio"], dtype=np.float32)
        audio_np = audio_np.reshape(data["audio_shape"])
        audio = torch.from_numpy(audio_np)

    return InferenceRequest(
        request_id=data["request_id"],
        request_type=RequestType(data["request_type"]),
        audio=audio,
        audio_duration=data["audio_duration"],
        prompt=data["prompt"],
        max_tokens=data["max_tokens"],
        temperature=data["temperature"],
        top_p=data["top_p"],
        top_k=data["top_k"],
        do_sample=data["do_sample"],
        repetition_penalty=data["repetition_penalty"],
        return_embeddings=data["return_embeddings"],
        priority=data["priority"],
        created_at=data["created_at"],
    )


def _serialize_result(result: InferenceResult) -> dict:
    """Serialize result for IPC."""
    return {
        "request_id": result.request_id,
        "request_type": result.request_type.value,
        "text": result.text,
        "tokens": result.tokens,
        "logprobs": result.logprobs,
        "finish_reason": result.finish_reason,
        "embedding": result.embedding.numpy().tobytes() if result.embedding is not None else None,
        "embedding_shape": list(result.embedding.shape) if result.embedding is not None else None,
        "frame_timestamps": (
            result.frame_timestamps.numpy().tobytes()
            if result.frame_timestamps is not None
            else None
        ),
        "frame_timestamps_shape": (
            list(result.frame_timestamps.shape)
            if result.frame_timestamps is not None
            else None
        ),
        "audio_duration": result.audio_duration,
        "processing_time": result.processing_time,
        "worker_id": result.worker_id,
        "error": result.error,
        "success": result.success,
    }


def _deserialize_result(data: dict) -> InferenceResult:
    """Deserialize result from IPC."""
    import numpy as np
    from flamingo_inference.executor.abstract import RequestType

    embedding = None
    if data["embedding"] is not None:
        embedding_np = np.frombuffer(data["embedding"], dtype=np.float32)
        embedding_np = embedding_np.reshape(data["embedding_shape"])
        embedding = torch.from_numpy(embedding_np)

    frame_timestamps = None
    if data["frame_timestamps"] is not None:
        ts_np = np.frombuffer(data["frame_timestamps"], dtype=np.float32)
        ts_np = ts_np.reshape(data["frame_timestamps_shape"])
        frame_timestamps = torch.from_numpy(ts_np)

    return InferenceResult(
        request_id=data["request_id"],
        request_type=RequestType(data["request_type"]),
        text=data["text"],
        tokens=data["tokens"],
        logprobs=data["logprobs"],
        finish_reason=data["finish_reason"],
        embedding=embedding,
        frame_timestamps=frame_timestamps,
        audio_duration=data["audio_duration"],
        processing_time=data["processing_time"],
        worker_id=data["worker_id"],
        error=data["error"],
        success=data["success"],
    )


class MultiprocExecutor(Executor):
    """Multi-process executor for multi-GPU data parallel deployment.

    Each GPU runs in a separate process with a full copy of the model.
    Requests are distributed using round-robin load balancing.
    """

    def __init__(self, config: FlamingoEngineConfig):
        """Initialize the executor.

        Args:
            config: Engine configuration
        """
        super().__init__(config)

        # Determine GPU IDs
        if config.executor.gpu_ids:
            self._gpu_ids = config.executor.gpu_ids
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("No CUDA device available")
            self._gpu_ids = list(range(torch.cuda.device_count()))

        # Worker processes and queues
        self._processes: list[mp.Process] = []
        self._request_queues: list[Queue] = []
        self._result_queues: list[Queue] = []
        self._ready_events: list[mp.Event] = []
        self._shutdown_events: list[mp.Event] = []

        # Round-robin counter
        self._next_worker = 0

        # Pending results tracking
        self._pending_requests: dict[str, int] = {}  # request_id -> worker_id

        logger.info(f"MultiprocExecutor: Will use GPUs {self._gpu_ids}")

    def start(self) -> None:
        """Start worker processes."""
        if self._is_running:
            logger.warning("Executor already running")
            return

        logger.info(f"MultiprocExecutor: Starting {len(self._gpu_ids)} workers")

        # Serialize config
        config_dict = self.config.to_dict()

        # Start worker processes
        for worker_id, gpu_id in enumerate(self._gpu_ids):
            request_queue = mp.Queue()
            result_queue = mp.Queue()
            ready_event = mp.Event()
            shutdown_event = mp.Event()

            process = mp.Process(
                target=_worker_process,
                args=(
                    worker_id,
                    gpu_id,
                    config_dict,
                    request_queue,
                    result_queue,
                    ready_event,
                    shutdown_event,
                ),
                daemon=True,
            )
            process.start()

            self._processes.append(process)
            self._request_queues.append(request_queue)
            self._result_queues.append(result_queue)
            self._ready_events.append(ready_event)
            self._shutdown_events.append(shutdown_event)

        # Wait for all workers to be ready
        logger.info("MultiprocExecutor: Waiting for workers to initialize...")
        timeout = 300  # 5 minutes
        for worker_id, event in enumerate(self._ready_events):
            if not event.wait(timeout=timeout):
                raise RuntimeError(f"Worker {worker_id} failed to start within {timeout}s")
            logger.info(f"Worker {worker_id}: Ready")

        self._is_running = True
        logger.info("MultiprocExecutor: All workers ready")

    def stop(self) -> None:
        """Stop all worker processes."""
        if not self._is_running:
            return

        logger.info("MultiprocExecutor: Stopping workers")

        # Signal shutdown
        for shutdown_event in self._shutdown_events:
            shutdown_event.set()

        # Send termination signal through queues
        for queue in self._request_queues:
            try:
                queue.put(None)
            except Exception:
                pass

        # Wait for processes to finish
        for process in self._processes:
            process.join(timeout=10)
            if process.is_alive():
                logger.warning(f"Force terminating process {process.pid}")
                process.terminate()

        # Clean up
        self._processes.clear()
        self._request_queues.clear()
        self._result_queues.clear()
        self._ready_events.clear()
        self._shutdown_events.clear()
        self._pending_requests.clear()

        self._is_running = False
        logger.info("MultiprocExecutor: Stopped")

    def execute(self, request: InferenceRequest) -> InferenceResult:
        """Execute a single request using round-robin distribution.

        Args:
            request: The inference request

        Returns:
            The inference result
        """
        if not self._is_running:
            return InferenceResult(
                request_id=request.request_id,
                request_type=request.request_type,
                error="Executor not running",
                success=False,
            )

        # Select worker (round-robin)
        worker_id = self._next_worker
        self._next_worker = (self._next_worker + 1) % len(self._gpu_ids)

        # Send request
        request_data = _serialize_request(request)
        self._request_queues[worker_id].put(request_data)
        self._pending_requests[request.request_id] = worker_id

        # Wait for result
        try:
            result_data = self._result_queues[worker_id].get(timeout=600)  # 10 min timeout
            result = _deserialize_result(result_data)
        except Exception as e:
            result = InferenceResult(
                request_id=request.request_id,
                request_type=request.request_type,
                error=f"Worker timeout or error: {e}",
                success=False,
            )

        del self._pending_requests[request.request_id]
        return result

    def execute_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        """Execute a batch of requests with parallel distribution.

        Distributes requests across workers and collects results.

        Args:
            requests: List of inference requests

        Returns:
            List of inference results
        """
        if not self._is_running:
            return [
                InferenceResult(
                    request_id=req.request_id,
                    request_type=req.request_type,
                    error="Executor not running",
                    success=False,
                )
                for req in requests
            ]

        # Distribute requests across workers
        request_to_worker: dict[str, int] = {}

        for i, request in enumerate(requests):
            worker_id = i % len(self._gpu_ids)
            request_data = _serialize_request(request)
            self._request_queues[worker_id].put(request_data)
            request_to_worker[request.request_id] = worker_id

        # Collect results
        results: dict[str, InferenceResult] = {}
        timeout = 600  # 10 min per request

        for _ in range(len(requests)):
            # Poll all result queues
            for worker_id, queue in enumerate(self._result_queues):
                try:
                    result_data = queue.get(timeout=timeout)
                    result = _deserialize_result(result_data)
                    results[result.request_id] = result
                    break
                except Exception:
                    continue

        # Order results by original request order
        ordered_results = []
        for request in requests:
            if request.request_id in results:
                ordered_results.append(results[request.request_id])
            else:
                ordered_results.append(
                    InferenceResult(
                        request_id=request.request_id,
                        request_type=request.request_type,
                        error="Result not received",
                        success=False,
                    )
                )

        return ordered_results

    @property
    def num_workers(self) -> int:
        """Get the number of active workers."""
        return len(self._gpu_ids) if self._is_running else 0

    @property
    def is_ready(self) -> bool:
        """Check if all workers are ready."""
        return (
            self._is_running
            and all(event.is_set() for event in self._ready_events)
        )

    def get_worker_stats(self) -> list[dict]:
        """Get statistics for each worker.

        Note: For multi-process executor, detailed stats require
        IPC which is not currently implemented. Returns basic info.
        """
        if not self._is_running:
            return []

        return [
            {
                "worker_id": i,
                "gpu_id": gpu_id,
                "is_ready": self._ready_events[i].is_set() if i < len(self._ready_events) else False,
                "process_alive": self._processes[i].is_alive() if i < len(self._processes) else False,
            }
            for i, gpu_id in enumerate(self._gpu_ids)
        ]
