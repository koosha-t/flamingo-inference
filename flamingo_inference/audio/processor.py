"""Audio preprocessing for Music Flamingo inference."""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    pass

from flamingo_inference.config import AudioConfig

logger = logging.getLogger(__name__)


@dataclass
class ProcessedAudio:
    """Processed audio ready for model input."""

    waveform: torch.Tensor  # Shape: (samples,) normalized mono audio
    sample_rate: int
    duration: float
    original_sample_rate: int
    was_resampled: bool
    was_normalized: bool


@dataclass
class AudioChunk:
    """A chunk of audio for processing long files."""

    waveform: torch.Tensor
    start_time: float
    end_time: float
    chunk_index: int
    total_chunks: int


class AudioProcessor:
    """Audio preprocessing pipeline for Music Flamingo.

    Handles:
    - Loading from files, bytes, or base64
    - Resampling to target sample rate (48kHz)
    - Mono conversion
    - Loudness normalization
    - Duration validation
    - Chunking for long audio
    """

    def __init__(self, config: AudioConfig):
        """Initialize the audio processor.

        Args:
            config: Audio processing configuration
        """
        self.config = config
        self._resampler_cache: dict[int, object] = {}

    def process(
        self,
        audio: torch.Tensor | np.ndarray | bytes | str | Path,
        sample_rate: int | None = None,
        format: str | None = None,
    ) -> ProcessedAudio:
        """Process audio from various input formats.

        Args:
            audio: Audio input - tensor, array, bytes, base64 string, or file path
            sample_rate: Sample rate (required for tensor/array input)
            format: Audio format hint for bytes input (e.g., "wav", "mp3")

        Returns:
            ProcessedAudio ready for model input
        """
        # Load audio if needed
        if isinstance(audio, (str, Path)):
            if isinstance(audio, str) and self._is_base64(audio):
                waveform, original_sr = self._load_from_base64(audio, format)
            else:
                waveform, original_sr = self._load_from_file(Path(audio))
        elif isinstance(audio, bytes):
            waveform, original_sr = self._load_from_bytes(audio, format)
        elif isinstance(audio, np.ndarray):
            if sample_rate is None:
                raise ValueError("sample_rate required for numpy array input")
            waveform = torch.from_numpy(audio).float()
            original_sr = sample_rate
        elif isinstance(audio, torch.Tensor):
            if sample_rate is None:
                raise ValueError("sample_rate required for tensor input")
            waveform = audio.float()
            original_sr = sample_rate
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

        # Convert to mono if stereo
        if waveform.dim() == 2:
            if waveform.shape[0] == 2:
                waveform = waveform.mean(dim=0)
            elif waveform.shape[1] == 2:
                waveform = waveform.mean(dim=1)
            else:
                waveform = waveform.squeeze()

        # Ensure 1D
        if waveform.dim() != 1:
            waveform = waveform.flatten()

        # Check duration
        duration = len(waveform) / original_sr
        if duration > self.config.max_duration:
            raise ValueError(
                f"Audio duration ({duration:.1f}s) exceeds maximum "
                f"({self.config.max_duration}s)"
            )

        # Resample if needed
        was_resampled = False
        if original_sr != self.config.sample_rate:
            waveform = self._resample(waveform, original_sr, self.config.sample_rate)
            was_resampled = True

        # Normalize if configured
        was_normalized = False
        if self.config.normalize:
            waveform = self._normalize(waveform)
            was_normalized = True

        # Recalculate duration after resampling
        duration = len(waveform) / self.config.sample_rate

        return ProcessedAudio(
            waveform=waveform,
            sample_rate=self.config.sample_rate,
            duration=duration,
            original_sample_rate=original_sr,
            was_resampled=was_resampled,
            was_normalized=was_normalized,
        )

    def chunk(self, audio: ProcessedAudio) -> list[AudioChunk]:
        """Split long audio into overlapping chunks.

        Args:
            audio: Processed audio to chunk

        Returns:
            List of AudioChunks
        """
        chunk_samples = int(self.config.chunk_duration * audio.sample_rate)
        overlap_samples = int(self.config.chunk_overlap * audio.sample_rate)
        step_samples = chunk_samples - overlap_samples

        waveform = audio.waveform
        total_samples = len(waveform)

        if total_samples <= chunk_samples:
            return [
                AudioChunk(
                    waveform=waveform,
                    start_time=0.0,
                    end_time=audio.duration,
                    chunk_index=0,
                    total_chunks=1,
                )
            ]

        chunks = []
        start = 0
        chunk_index = 0

        # Calculate total chunks
        total_chunks = max(1, (total_samples - overlap_samples) // step_samples)
        if (total_samples - overlap_samples) % step_samples > 0:
            total_chunks += 1

        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            chunk_waveform = waveform[start:end]

            # Pad last chunk if too short
            if len(chunk_waveform) < chunk_samples // 2:
                # Skip very short final chunks
                break

            start_time = start / audio.sample_rate
            end_time = end / audio.sample_rate

            chunks.append(
                AudioChunk(
                    waveform=chunk_waveform,
                    start_time=start_time,
                    end_time=end_time,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                )
            )

            start += step_samples
            chunk_index += 1

        # Update total_chunks in all chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _load_from_file(self, path: Path) -> tuple[torch.Tensor, int]:
        """Load audio from file."""
        import torchaudio

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        waveform, sample_rate = torchaudio.load(str(path))
        return waveform, sample_rate

    def _load_from_bytes(
        self, data: bytes, format: str | None = None
    ) -> tuple[torch.Tensor, int]:
        """Load audio from bytes."""
        import soundfile as sf

        audio_io = io.BytesIO(data)
        audio_data, sample_rate = sf.read(audio_io, dtype="float32")
        waveform = torch.from_numpy(audio_data)
        return waveform, sample_rate

    def _load_from_base64(
        self, data: str, format: str | None = None
    ) -> tuple[torch.Tensor, int]:
        """Load audio from base64 string."""
        decoded = base64.b64decode(data)
        return self._load_from_bytes(decoded, format)

    def _is_base64(self, s: str) -> bool:
        """Check if string is base64 encoded."""
        # Simple heuristic: check if it looks like a file path
        if "/" in s or "\\" in s or s.endswith((".wav", ".mp3", ".flac", ".ogg")):
            return False
        try:
            # Try to decode first 100 chars
            base64.b64decode(s[:100] + "==")
            return True
        except Exception:
            return False

    def _resample(
        self, waveform: torch.Tensor, orig_sr: int, target_sr: int
    ) -> torch.Tensor:
        """Resample audio to target sample rate."""
        import torchaudio.transforms as T

        cache_key = (orig_sr, target_sr)
        if cache_key not in self._resampler_cache:
            self._resampler_cache[cache_key] = T.Resample(orig_sr, target_sr)

        resampler = self._resampler_cache[cache_key]

        # Add batch dimension if needed
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            waveform = resampler(waveform)
            waveform = waveform.squeeze(0)
        else:
            waveform = resampler(waveform)

        return waveform

    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio loudness.

        Uses simple peak normalization. For LUFS normalization,
        additional dependencies would be needed.
        """
        # Peak normalization to -1dB
        peak = waveform.abs().max()
        if peak > 0:
            target_peak = 10 ** (-1 / 20)  # -1 dB
            waveform = waveform * (target_peak / peak)
        return waveform

    def estimate_memory(self, audio: ProcessedAudio) -> int:
        """Estimate memory required for processing this audio.

        Args:
            audio: Processed audio

        Returns:
            Estimated memory in bytes
        """
        # Audio tensor memory
        audio_bytes = audio.waveform.numel() * 4  # float32

        # Rough estimate of model processing memory
        # Based on empirical observation: ~100MB base + 10MB per second of audio
        base_memory = 100 * 1024 * 1024  # 100 MB
        per_second_memory = 10 * 1024 * 1024  # 10 MB

        return audio_bytes + base_memory + int(audio.duration * per_second_memory)


def load_audio(
    path: str | Path,
    config: AudioConfig | None = None,
) -> ProcessedAudio:
    """Convenience function to load and process audio.

    Args:
        path: Path to audio file
        config: Audio configuration (uses defaults if None)

    Returns:
        ProcessedAudio ready for model input
    """
    if config is None:
        config = AudioConfig()
    processor = AudioProcessor(config)
    return processor.process(path)
