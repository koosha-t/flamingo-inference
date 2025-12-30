"""Health check utilities for Music Flamingo server."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from flamingo_inference.engine import AsyncFlamingoEngine

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str | None = None
    latency_ms: float | None = None
    details: dict | None = None


@dataclass
class HealthReport:
    """Complete health report."""

    status: HealthStatus
    checks: list[HealthCheck] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def is_healthy(self) -> bool:
        """Check if overall status is healthy."""
        return self.status == HealthStatus.HEALTHY

    @property
    def is_ready(self) -> bool:
        """Check if service is ready to accept requests."""
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "latency_ms": c.latency_ms,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


class HealthChecker:
    """Health checker for the inference server.

    Performs various health checks:
    - Engine readiness
    - GPU availability
    - Model loaded
    - Queue health
    """

    def __init__(self, engine: "AsyncFlamingoEngine"):
        """Initialize the health checker.

        Args:
            engine: The engine to check
        """
        self.engine = engine
        self._last_check: HealthReport | None = None
        self._check_interval = 10.0  # seconds

    async def check(self) -> HealthReport:
        """Perform all health checks.

        Returns:
            Complete health report
        """
        checks = []

        # Engine check
        checks.append(await self._check_engine())

        # GPU check
        checks.append(await self._check_gpu())

        # Scheduler check
        checks.append(await self._check_scheduler())

        # Determine overall status
        statuses = [c.status for c in checks]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        else:
            overall = HealthStatus.DEGRADED

        report = HealthReport(status=overall, checks=checks)
        self._last_check = report
        return report

    async def _check_engine(self) -> HealthCheck:
        """Check engine status."""
        start = time.time()

        if not self.engine.is_running:
            return HealthCheck(
                name="engine",
                status=HealthStatus.UNHEALTHY,
                message="Engine not running",
                latency_ms=(time.time() - start) * 1000,
            )

        if not self.engine.is_ready:
            return HealthCheck(
                name="engine",
                status=HealthStatus.DEGRADED,
                message="Engine not ready",
                latency_ms=(time.time() - start) * 1000,
            )

        return HealthCheck(
            name="engine",
            status=HealthStatus.HEALTHY,
            message="Engine running and ready",
            latency_ms=(time.time() - start) * 1000,
        )

    async def _check_gpu(self) -> HealthCheck:
        """Check GPU availability."""
        import torch

        start = time.time()

        if not torch.cuda.is_available():
            return HealthCheck(
                name="gpu",
                status=HealthStatus.UNHEALTHY,
                message="No GPU available",
                latency_ms=(time.time() - start) * 1000,
            )

        try:
            # Get GPU info
            device_count = torch.cuda.device_count()
            devices = []
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_total = props.total_memory
                devices.append({
                    "id": i,
                    "name": props.name,
                    "memory_used_gb": memory_allocated / 1e9,
                    "memory_total_gb": memory_total / 1e9,
                    "utilization": memory_allocated / memory_total,
                })

            return HealthCheck(
                name="gpu",
                status=HealthStatus.HEALTHY,
                message=f"{device_count} GPU(s) available",
                latency_ms=(time.time() - start) * 1000,
                details={"devices": devices},
            )

        except Exception as e:
            return HealthCheck(
                name="gpu",
                status=HealthStatus.DEGRADED,
                message=f"GPU check failed: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    async def _check_scheduler(self) -> HealthCheck:
        """Check scheduler status."""
        start = time.time()

        try:
            stats = self.engine.get_stats()
            scheduler_stats = stats.get("scheduler", {})

            queue_size = scheduler_stats.get("current_queue_size", 0)
            max_queue = self.engine.config.scheduler.max_waiting_requests

            if queue_size >= max_queue * 0.9:
                return HealthCheck(
                    name="scheduler",
                    status=HealthStatus.DEGRADED,
                    message=f"Queue near capacity: {queue_size}/{max_queue}",
                    latency_ms=(time.time() - start) * 1000,
                    details=scheduler_stats,
                )

            return HealthCheck(
                name="scheduler",
                status=HealthStatus.HEALTHY,
                message=f"Queue healthy: {queue_size}/{max_queue}",
                latency_ms=(time.time() - start) * 1000,
                details=scheduler_stats,
            )

        except Exception as e:
            return HealthCheck(
                name="scheduler",
                status=HealthStatus.UNHEALTHY,
                message=f"Scheduler check failed: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    async def liveness(self) -> bool:
        """Simple liveness check.

        Returns:
            True if the service is alive
        """
        return True

    async def readiness(self) -> bool:
        """Check if service is ready to accept requests.

        Returns:
            True if ready
        """
        return self.engine.is_running and self.engine.is_ready

    @property
    def last_check(self) -> HealthReport | None:
        """Get the last health check report."""
        return self._last_check
