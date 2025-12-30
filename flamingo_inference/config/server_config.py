"""Server configuration for the FastAPI server."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CORSConfig:
    """CORS configuration."""

    enabled: bool = True
    allow_origins: list[str] = field(default_factory=lambda: ["*"])
    allow_methods: list[str] = field(default_factory=lambda: ["*"])
    allow_headers: list[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = False


@dataclass
class MetricsConfig:
    """Prometheus metrics configuration."""

    enabled: bool = True
    port: int | None = None  # None = same as server port
    path: str = "/metrics"


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    json_format: bool = False
    access_log: bool = True


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    enabled: bool = False
    requests_per_minute: int = 60
    burst_size: int = 10


@dataclass
class BatchJobConfig:
    """Batch job processing configuration."""

    enabled: bool = True
    max_concurrent_jobs: int = 10
    job_timeout_seconds: int = 3600  # 1 hour
    result_retention_seconds: int = 86400  # 24 hours
    storage_dir: Path | None = None


@dataclass
class ServerConfig:
    """Complete configuration for the API server."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    timeout_keep_alive: int = 30
    max_request_size_mb: int = 100  # Max audio upload size

    cors: CORSConfig = field(default_factory=CORSConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    batch_job: BatchJobConfig = field(default_factory=BatchJobConfig)

    # API settings
    api_key: str | None = None  # None = no authentication
    allowed_models: list[str] = field(
        default_factory=lambda: ["nvidia/music-flamingo-hf"]
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ServerConfig:
        """Load configuration from YAML file."""
        import yaml

        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data.get("server", data))

    @classmethod
    def _from_dict(cls, data: dict) -> ServerConfig:
        """Create config from dictionary."""
        # Create a default instance to get default values
        defaults = cls()

        cors_data = data.get("cors", {})
        metrics_data = data.get("metrics", {})
        logging_data = data.get("logging", {})
        rate_limit_data = data.get("rate_limit", {})
        batch_job_data = data.get("batch_job", {})

        if "storage_dir" in batch_job_data and batch_job_data["storage_dir"]:
            batch_job_data["storage_dir"] = Path(batch_job_data["storage_dir"])

        return cls(
            host=data.get("host", defaults.host),
            port=data.get("port", defaults.port),
            workers=data.get("workers", defaults.workers),
            timeout_keep_alive=data.get("timeout_keep_alive", defaults.timeout_keep_alive),
            max_request_size_mb=data.get("max_request_size_mb", defaults.max_request_size_mb),
            cors=CORSConfig(**cors_data),
            metrics=MetricsConfig(**metrics_data),
            logging=LoggingConfig(**logging_data),
            rate_limit=RateLimitConfig(**rate_limit_data),
            batch_job=BatchJobConfig(**batch_job_data),
            api_key=data.get("api_key"),
            allowed_models=data.get("allowed_models", defaults.allowed_models),
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "timeout_keep_alive": self.timeout_keep_alive,
            "max_request_size_mb": self.max_request_size_mb,
            "cors": {
                "enabled": self.cors.enabled,
                "allow_origins": self.cors.allow_origins,
                "allow_methods": self.cors.allow_methods,
                "allow_headers": self.cors.allow_headers,
                "allow_credentials": self.cors.allow_credentials,
            },
            "metrics": {
                "enabled": self.metrics.enabled,
                "port": self.metrics.port,
                "path": self.metrics.path,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "json_format": self.logging.json_format,
                "access_log": self.logging.access_log,
            },
            "rate_limit": {
                "enabled": self.rate_limit.enabled,
                "requests_per_minute": self.rate_limit.requests_per_minute,
                "burst_size": self.rate_limit.burst_size,
            },
            "batch_job": {
                "enabled": self.batch_job.enabled,
                "max_concurrent_jobs": self.batch_job.max_concurrent_jobs,
                "job_timeout_seconds": self.batch_job.job_timeout_seconds,
                "result_retention_seconds": self.batch_job.result_retention_seconds,
                "storage_dir": (
                    str(self.batch_job.storage_dir)
                    if self.batch_job.storage_dir
                    else None
                ),
            },
            "api_key": self.api_key,
            "allowed_models": self.allowed_models,
        }
