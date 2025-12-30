"""Executor module for model inference."""

from flamingo_inference.executor.abstract import (
    Executor,
    InferenceRequest,
    InferenceResult,
    RequestStatus,
    RequestType,
)
from flamingo_inference.executor.multiproc_executor import MultiprocExecutor
from flamingo_inference.executor.uniproc_executor import UniprocExecutor
from flamingo_inference.executor.worker import FlamingoWorker, WorkerStats

__all__ = [
    # Abstract
    "Executor",
    "InferenceRequest",
    "InferenceResult",
    "RequestType",
    "RequestStatus",
    # Executors
    "UniprocExecutor",
    "MultiprocExecutor",
    # Worker
    "FlamingoWorker",
    "WorkerStats",
]


def create_executor(config) -> Executor:
    """Create an executor based on configuration.

    Args:
        config: Engine configuration

    Returns:
        Configured executor instance
    """
    from flamingo_inference.config import ExecutorType

    if config.executor.type == ExecutorType.UNIPROC:
        return UniprocExecutor(config)
    elif config.executor.type == ExecutorType.MULTIPROC:
        return MultiprocExecutor(config)
    else:
        raise ValueError(f"Unknown executor type: {config.executor.type}")
