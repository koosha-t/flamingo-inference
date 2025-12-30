"""Engine configuration for Music Flamingo inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

import torch


class ExecutorType(str, Enum):
    """Executor type for model inference."""

    UNIPROC = "uniproc"  # Single GPU, single process
    MULTIPROC = "multiproc"  # Multi-GPU, data parallel


class SchedulerPolicy(str, Enum):
    """Scheduling policy for requests."""

    FCFS = "fcfs"  # First come, first served
    PRIORITY = "priority"  # Priority-based scheduling


@dataclass
class ModelConfig:
    """Configuration for the Music Flamingo model."""

    name: str = "nvidia/music-flamingo-hf"
    revision: str | None = None
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    torch_compile: bool = False
    max_memory: dict[int, str] | None = None  # Per-GPU memory limits

    @property
    def torch_dtype(self) -> torch.dtype:
        """Get torch dtype from string."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map[self.dtype]


@dataclass
class AudioConfig:
    """Configuration for audio processing."""

    sample_rate: int = 48000  # Music Flamingo expects 48kHz
    max_duration: float = 600.0  # 10 minutes max
    chunk_duration: float = 30.0  # Chunk duration for long audio
    chunk_overlap: float = 5.0  # Overlap between chunks
    normalize: bool = True
    target_loudness: float = -23.0  # LUFS for normalization


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_beams: int = 1


@dataclass
class SchedulerConfig:
    """Configuration for the request scheduler."""

    max_batch_size: int = 8
    max_waiting_requests: int = 1000
    max_total_audio_duration: float = 300.0  # Max total audio in a batch (seconds)
    policy: SchedulerPolicy = SchedulerPolicy.PRIORITY
    batch_wait_timeout: float = 0.05  # Seconds to wait for batching


@dataclass
class ExecutorConfig:
    """Configuration for the executor."""

    type: ExecutorType = ExecutorType.UNIPROC
    gpu_ids: list[int] | None = None  # None = auto-detect
    num_workers: int | None = None  # None = one per GPU


@dataclass
class CacheConfig:
    """Configuration for embedding cache."""

    enabled: bool = True
    max_size_gb: float = 10.0
    cache_dir: Path | None = None
    ttl_seconds: int = 86400  # 24 hours


@dataclass
class FlamingoEngineConfig:
    """Complete configuration for the Flamingo inference engine."""

    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    # IPC settings
    ipc_path: str = "ipc:///tmp/flamingo_engine_{pid}"

    @classmethod
    def from_yaml(cls, path: str | Path) -> FlamingoEngineConfig:
        """Load configuration from YAML file."""
        import yaml

        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> FlamingoEngineConfig:
        """Create config from dictionary."""
        model_data = data.get("model", {})
        audio_data = data.get("audio", {})
        generation_data = data.get("generation", {})
        scheduler_data = data.get("scheduler", {})
        executor_data = data.get("executor", {})
        cache_data = data.get("cache", {})

        # Convert string enums
        if "policy" in scheduler_data:
            scheduler_data["policy"] = SchedulerPolicy(scheduler_data["policy"])
        if "type" in executor_data:
            executor_data["type"] = ExecutorType(executor_data["type"])
        if "cache_dir" in cache_data and cache_data["cache_dir"]:
            cache_data["cache_dir"] = Path(cache_data["cache_dir"])

        return cls(
            model=ModelConfig(**model_data),
            audio=AudioConfig(**audio_data),
            generation=GenerationConfig(**generation_data),
            scheduler=SchedulerConfig(**scheduler_data),
            executor=ExecutorConfig(**executor_data),
            cache=CacheConfig(**cache_data),
            ipc_path=data.get("ipc_path", cls.ipc_path),
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "model": {
                "name": self.model.name,
                "revision": self.model.revision,
                "dtype": self.model.dtype,
                "device_map": self.model.device_map,
                "trust_remote_code": self.model.trust_remote_code,
                "torch_compile": self.model.torch_compile,
                "max_memory": self.model.max_memory,
            },
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "max_duration": self.audio.max_duration,
                "chunk_duration": self.audio.chunk_duration,
                "chunk_overlap": self.audio.chunk_overlap,
                "normalize": self.audio.normalize,
                "target_loudness": self.audio.target_loudness,
            },
            "generation": {
                "max_tokens": self.generation.max_tokens,
                "temperature": self.generation.temperature,
                "top_p": self.generation.top_p,
                "top_k": self.generation.top_k,
                "repetition_penalty": self.generation.repetition_penalty,
                "do_sample": self.generation.do_sample,
                "num_beams": self.generation.num_beams,
            },
            "scheduler": {
                "max_batch_size": self.scheduler.max_batch_size,
                "max_waiting_requests": self.scheduler.max_waiting_requests,
                "max_total_audio_duration": self.scheduler.max_total_audio_duration,
                "policy": self.scheduler.policy.value,
                "batch_wait_timeout": self.scheduler.batch_wait_timeout,
            },
            "executor": {
                "type": self.executor.type.value,
                "gpu_ids": self.executor.gpu_ids,
                "num_workers": self.executor.num_workers,
            },
            "cache": {
                "enabled": self.cache.enabled,
                "max_size_gb": self.cache.max_size_gb,
                "cache_dir": str(self.cache.cache_dir) if self.cache.cache_dir else None,
                "ttl_seconds": self.cache.ttl_seconds,
            },
            "ipc_path": self.ipc_path,
        }
