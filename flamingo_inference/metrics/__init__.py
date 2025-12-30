"""Metrics module for Music Flamingo server."""

from flamingo_inference.metrics.prometheus import (
    init_model_info,
    record_generation,
    record_queue_wait,
    setup_metrics,
    track_request,
    update_from_engine_stats,
    update_gpu_metrics,
)

__all__ = [
    "setup_metrics",
    "track_request",
    "record_generation",
    "record_queue_wait",
    "update_gpu_metrics",
    "update_from_engine_stats",
    "init_model_info",
]
