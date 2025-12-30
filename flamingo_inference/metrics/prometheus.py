"""Prometheus metrics for Music Flamingo server."""

from __future__ import annotations

import time
from functools import wraps
from typing import TYPE_CHECKING, Callable

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

if TYPE_CHECKING:
    from fastapi import Request, Response


# ============================================================================
# Request Metrics
# ============================================================================

REQUEST_COUNT = Counter(
    "flamingo_requests_total",
    "Total number of requests",
    ["endpoint", "method", "status"],
)

REQUEST_LATENCY = Histogram(
    "flamingo_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint", "method"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

REQUEST_IN_PROGRESS = Gauge(
    "flamingo_requests_in_progress",
    "Number of requests currently in progress",
    ["endpoint"],
)

# ============================================================================
# Audio Metrics
# ============================================================================

AUDIO_DURATION = Histogram(
    "flamingo_audio_duration_seconds",
    "Audio duration in seconds",
    ["endpoint"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

AUDIO_PROCESSED = Counter(
    "flamingo_audio_processed_seconds_total",
    "Total audio processed in seconds",
    ["endpoint"],
)

# ============================================================================
# Generation Metrics
# ============================================================================

GENERATION_TOKENS = Histogram(
    "flamingo_generation_tokens",
    "Number of tokens generated",
    ["endpoint"],
    buckets=[10, 50, 100, 200, 500, 1000, 2000, 4000],
)

GENERATION_LATENCY = Histogram(
    "flamingo_generation_latency_seconds",
    "Generation latency in seconds",
    ["request_type"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
)

# ============================================================================
# Queue Metrics
# ============================================================================

QUEUE_SIZE = Gauge(
    "flamingo_queue_size",
    "Current number of requests in queue",
)

QUEUE_WAIT_TIME = Histogram(
    "flamingo_queue_wait_seconds",
    "Time spent waiting in queue",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# ============================================================================
# Worker Metrics
# ============================================================================

WORKER_COUNT = Gauge(
    "flamingo_workers_total",
    "Total number of workers",
)

WORKER_ACTIVE = Gauge(
    "flamingo_workers_active",
    "Number of active workers",
)

WORKER_REQUESTS = Counter(
    "flamingo_worker_requests_total",
    "Total requests processed by workers",
    ["worker_id"],
)

WORKER_ERRORS = Counter(
    "flamingo_worker_errors_total",
    "Total errors by workers",
    ["worker_id"],
)

# ============================================================================
# GPU Metrics
# ============================================================================

GPU_MEMORY_USED = Gauge(
    "flamingo_gpu_memory_used_bytes",
    "GPU memory used in bytes",
    ["gpu_id"],
)

GPU_MEMORY_TOTAL = Gauge(
    "flamingo_gpu_memory_total_bytes",
    "GPU memory total in bytes",
    ["gpu_id"],
)

GPU_UTILIZATION = Gauge(
    "flamingo_gpu_utilization_percent",
    "GPU utilization percentage",
    ["gpu_id"],
)

# ============================================================================
# Model Info
# ============================================================================

MODEL_INFO = Info(
    "flamingo_model",
    "Model information",
)


# ============================================================================
# Metrics Collection
# ============================================================================


def update_gpu_metrics() -> None:
    """Update GPU metrics from CUDA."""
    import torch

    if not torch.cuda.is_available():
        return

    for i in range(torch.cuda.device_count()):
        try:
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory

            GPU_MEMORY_USED.labels(gpu_id=str(i)).set(memory_allocated)
            GPU_MEMORY_TOTAL.labels(gpu_id=str(i)).set(memory_total)
            GPU_UTILIZATION.labels(gpu_id=str(i)).set(
                (memory_allocated / memory_total) * 100 if memory_total > 0 else 0
            )
        except Exception:
            pass


def update_from_engine_stats(stats: dict) -> None:
    """Update metrics from engine statistics.

    Args:
        stats: Engine statistics dictionary
    """
    # Scheduler metrics
    scheduler = stats.get("scheduler", {})
    QUEUE_SIZE.set(scheduler.get("current_queue_size", 0))

    # Worker metrics
    executor = stats.get("executor", {})
    workers = executor.get("workers", [])
    WORKER_COUNT.set(len(workers))
    WORKER_ACTIVE.set(sum(1 for w in workers if w.get("is_ready", False)))

    for worker in workers:
        worker_id = str(worker.get("worker_id", 0))
        # Note: These are cumulative, so we'd need to track deltas
        # For now, just update GPU metrics from worker stats
        if "gpu_memory_allocated_gb" in worker:
            GPU_MEMORY_USED.labels(gpu_id=worker_id).set(
                worker["gpu_memory_allocated_gb"] * 1e9
            )


# ============================================================================
# Request Tracking Decorator
# ============================================================================


def track_request(endpoint: str):
    """Decorator to track request metrics.

    Args:
        endpoint: Endpoint name for labeling
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            REQUEST_IN_PROGRESS.labels(endpoint=endpoint).inc()
            start_time = time.time()

            try:
                response = await func(*args, **kwargs)
                status = "success"
                return response
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                REQUEST_IN_PROGRESS.labels(endpoint=endpoint).dec()
                REQUEST_LATENCY.labels(endpoint=endpoint, method="POST").observe(duration)
                REQUEST_COUNT.labels(
                    endpoint=endpoint, method="POST", status=status
                ).inc()

        return wrapper
    return decorator


def record_generation(
    request_type: str,
    audio_duration: float,
    num_tokens: int,
    latency: float,
    endpoint: str = "generate",
) -> None:
    """Record generation metrics.

    Args:
        request_type: Type of request (generate, embed, analyze)
        audio_duration: Audio duration in seconds
        num_tokens: Number of tokens generated
        latency: Generation latency in seconds
        endpoint: Endpoint name
    """
    AUDIO_DURATION.labels(endpoint=endpoint).observe(audio_duration)
    AUDIO_PROCESSED.labels(endpoint=endpoint).inc(audio_duration)
    GENERATION_TOKENS.labels(endpoint=endpoint).observe(num_tokens)
    GENERATION_LATENCY.labels(request_type=request_type).observe(latency)


def record_queue_wait(wait_time: float) -> None:
    """Record queue wait time.

    Args:
        wait_time: Time spent waiting in seconds
    """
    QUEUE_WAIT_TIME.observe(wait_time)


# ============================================================================
# FastAPI Integration
# ============================================================================


def setup_metrics(app) -> None:
    """Setup Prometheus metrics endpoint for FastAPI.

    Args:
        app: FastAPI application
    """
    from fastapi import Response

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        # Update GPU metrics before returning
        update_gpu_metrics()

        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )


def init_model_info(model_name: str, dtype: str, device: str) -> None:
    """Initialize model info metric.

    Args:
        model_name: Model name
        dtype: Data type
        device: Device
    """
    MODEL_INFO.info({
        "model_name": model_name,
        "dtype": dtype,
        "device": device,
    })
