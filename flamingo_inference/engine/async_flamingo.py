"""Async Flamingo Engine - main entry point for inference."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import AsyncIterator, Union

import torch

from flamingo_inference.audio import AudioProcessor, ProcessedAudio
from flamingo_inference.config import FlamingoEngineConfig
from flamingo_inference.executor import (
    InferenceRequest,
    InferenceResult,
    RequestType,
    create_executor,
)
from flamingo_inference.scheduler import FlamingoScheduler

logger = logging.getLogger(__name__)


AudioInput = Union[torch.Tensor, bytes, str, Path]


class AsyncFlamingoEngine:
    """Async inference engine for Music Flamingo.

    This is the main entry point for running inference. It manages:
    - Audio preprocessing
    - Request lifecycle
    - Executor coordination
    - Result delivery

    Example:
        ```python
        config = FlamingoEngineConfig.from_yaml("config.yaml")
        engine = AsyncFlamingoEngine(config)

        async with engine:
            result = await engine.generate(
                audio="song.wav",
                prompt="Describe this music in detail.",
            )
            print(result.text)
        ```
    """

    def __init__(self, config: FlamingoEngineConfig):
        """Initialize the engine.

        Args:
            config: Engine configuration
        """
        self.config = config
        self._audio_processor = AudioProcessor(config.audio)
        self._executor = create_executor(config)
        self._scheduler: FlamingoScheduler | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._is_running = False

    async def start(self) -> None:
        """Start the engine."""
        if self._is_running:
            logger.warning("Engine already running")
            return

        logger.info("Starting AsyncFlamingoEngine...")

        # Get event loop
        self._loop = asyncio.get_running_loop()

        # Start executor
        logger.info("Starting executor...")
        await asyncio.get_running_loop().run_in_executor(
            None, self._executor.start
        )

        # Create and start scheduler
        self._scheduler = FlamingoScheduler(
            config=self.config.scheduler,
            executor_callback=self._executor.execute_batch,
        )
        self._scheduler.start(loop=self._loop)

        self._is_running = True
        logger.info("AsyncFlamingoEngine started")

    async def stop(self) -> None:
        """Stop the engine."""
        if not self._is_running:
            return

        logger.info("Stopping AsyncFlamingoEngine...")

        # Stop scheduler
        if self._scheduler is not None:
            self._scheduler.stop()
            self._scheduler = None

        # Stop executor
        await asyncio.get_running_loop().run_in_executor(
            None, self._executor.stop
        )

        self._is_running = False
        logger.info("AsyncFlamingoEngine stopped")

    async def generate(
        self,
        audio: AudioInput,
        prompt: str,
        sample_rate: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        do_sample: bool | None = None,
        repetition_penalty: float | None = None,
        priority: int = 0,
    ) -> InferenceResult:
        """Generate text from audio input.

        Args:
            audio: Audio input (file path, bytes, or tensor)
            prompt: Text prompt/instruction
            sample_rate: Sample rate (required if audio is tensor)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            do_sample: Whether to use sampling
            repetition_penalty: Repetition penalty
            priority: Request priority (higher = more priority)

        Returns:
            InferenceResult with generated text
        """
        self._check_running()

        # Process audio
        processed = await self._process_audio(audio, sample_rate)

        # Create request
        request = InferenceRequest(
            request_type=RequestType.GENERATE,
            audio=processed.waveform,
            audio_duration=processed.duration,
            prompt=prompt,
            max_tokens=max_tokens or self.config.generation.max_tokens,
            temperature=temperature if temperature is not None else self.config.generation.temperature,
            top_p=top_p if top_p is not None else self.config.generation.top_p,
            top_k=top_k if top_k is not None else self.config.generation.top_k,
            do_sample=do_sample if do_sample is not None else self.config.generation.do_sample,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else self.config.generation.repetition_penalty,
            priority=priority,
        )

        # Submit and wait
        return await self._scheduler.submit(request)

    async def embed(
        self,
        audio: AudioInput,
        sample_rate: int | None = None,
        priority: int = 0,
    ) -> InferenceResult:
        """Extract audio embeddings.

        Args:
            audio: Audio input (file path, bytes, or tensor)
            sample_rate: Sample rate (required if audio is tensor)
            priority: Request priority

        Returns:
            InferenceResult with embedding tensor
        """
        self._check_running()

        # Process audio
        processed = await self._process_audio(audio, sample_rate)

        # Create request
        request = InferenceRequest(
            request_type=RequestType.EMBED,
            audio=processed.waveform,
            audio_duration=processed.duration,
            return_embeddings=True,
            priority=priority,
        )

        # Submit and wait
        return await self._scheduler.submit(request)

    async def analyze(
        self,
        audio: AudioInput,
        prompt: str | None = None,
        sample_rate: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        priority: int = 0,
    ) -> InferenceResult:
        """Generate both caption and embeddings.

        Args:
            audio: Audio input (file path, bytes, or tensor)
            prompt: Text prompt (uses default if not provided)
            sample_rate: Sample rate (required if audio is tensor)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            priority: Request priority

        Returns:
            InferenceResult with text and embeddings
        """
        self._check_running()

        # Process audio
        processed = await self._process_audio(audio, sample_rate)

        # Default prompt for analysis
        if prompt is None:
            prompt = (
                "Provide a detailed description of this music, including "
                "genre, mood, instrumentation, tempo, and any notable characteristics."
            )

        # Create request
        request = InferenceRequest(
            request_type=RequestType.ANALYZE,
            audio=processed.waveform,
            audio_duration=processed.duration,
            prompt=prompt,
            max_tokens=max_tokens or self.config.generation.max_tokens,
            temperature=temperature if temperature is not None else self.config.generation.temperature,
            top_p=self.config.generation.top_p,
            top_k=self.config.generation.top_k,
            do_sample=self.config.generation.do_sample,
            repetition_penalty=self.config.generation.repetition_penalty,
            return_embeddings=True,
            priority=priority,
        )

        # Submit and wait
        return await self._scheduler.submit(request)

    async def caption(
        self,
        audio: AudioInput,
        style: str = "detailed",
        sample_rate: int | None = None,
        max_tokens: int | None = None,
        priority: int = 0,
    ) -> str:
        """Generate a caption for the audio.

        Convenience method that returns just the caption text.

        Args:
            audio: Audio input
            style: Caption style ("detailed", "brief", "technical")
            sample_rate: Sample rate
            max_tokens: Maximum tokens
            priority: Request priority

        Returns:
            Caption text
        """
        # Select prompt based on style
        prompts = {
            "detailed": (
                "Provide a comprehensive description of this music. Include details about "
                "the genre, mood, instrumentation, tempo, melody, harmony, and any "
                "notable production techniques or stylistic elements."
            ),
            "brief": "Briefly describe this music in 2-3 sentences.",
            "technical": (
                "Analyze this music from a technical perspective. Describe the "
                "time signature, key, chord progressions, instrumentation, "
                "production techniques, and audio quality."
            ),
        }
        prompt = prompts.get(style, prompts["detailed"])

        result = await self.generate(
            audio=audio,
            prompt=prompt,
            sample_rate=sample_rate,
            max_tokens=max_tokens,
            priority=priority,
        )

        if not result.success:
            raise RuntimeError(f"Caption generation failed: {result.error}")

        return result.text or ""

    async def _process_audio(
        self,
        audio: AudioInput,
        sample_rate: int | None,
    ) -> ProcessedAudio:
        """Process audio input in executor to avoid blocking."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._audio_processor.process,
            audio,
            sample_rate,
            None,  # format
        )

    def _check_running(self) -> None:
        """Check if engine is running."""
        if not self._is_running:
            raise RuntimeError("Engine not running. Call start() first.")
        if self._scheduler is None:
            raise RuntimeError("Scheduler not initialized")

    @property
    def is_running(self) -> bool:
        """Check if the engine is running."""
        return self._is_running

    @property
    def is_ready(self) -> bool:
        """Check if the engine is ready to accept requests."""
        return self._is_running and self._executor.is_ready

    def get_stats(self) -> dict:
        """Get engine statistics."""
        stats = {
            "is_running": self._is_running,
            "is_ready": self.is_ready,
            "executor": {
                "num_workers": self._executor.num_workers,
                "workers": self._executor.get_worker_stats(),
            },
        }

        if self._scheduler is not None:
            stats["scheduler"] = self._scheduler.get_stats()

        return stats

    async def __aenter__(self) -> "AsyncFlamingoEngine":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async context manager exit."""
        await self.stop()
        return False


# Convenience function for one-off inference
async def quick_caption(
    audio: AudioInput,
    config: FlamingoEngineConfig | None = None,
    style: str = "detailed",
) -> str:
    """Quick caption generation without managing engine lifecycle.

    Args:
        audio: Audio input
        config: Engine config (uses defaults if None)
        style: Caption style

    Returns:
        Caption text
    """
    if config is None:
        config = FlamingoEngineConfig()

    async with AsyncFlamingoEngine(config) as engine:
        return await engine.caption(audio, style=style)
