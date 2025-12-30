"""Music Flamingo model wrapper for inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

from flamingo_inference.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """Output from text generation."""

    text: str
    tokens: list[int]
    logprobs: list[float] | None = None
    finish_reason: str = "stop"


@dataclass
class EmbeddingOutput:
    """Output from embedding extraction."""

    embedding: torch.Tensor  # Shape: (num_frames, embedding_dim)
    frame_timestamps: torch.Tensor  # Shape: (num_frames,)
    audio_duration: float


class FlamingoModel:
    """Wrapper for Music Flamingo model with generation and embedding extraction.

    This class wraps the HuggingFace AudioFlamingo3ForConditionalGeneration model
    and provides methods for:
    - Text generation (captions, Q&A)
    - Audio embedding extraction (AF-Whisper encoder)
    """

    EMBEDDING_DIM = 1536  # AF-Whisper embedding dimension
    FRAME_RATE = 50  # Frames per second from audio encoder

    def __init__(
        self,
        model: PreTrainedModel,
        processor: ProcessorMixin,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        """Initialize the model wrapper.

        Args:
            model: Loaded AudioFlamingo3ForConditionalGeneration model
            processor: Loaded AutoProcessor
            config: Model configuration
            device: Target device (if None, uses model's device)
        """
        self.model = model
        self.processor = processor
        self.config = config
        self._device = device or next(model.parameters()).device

        # Compile model if requested
        if config.torch_compile:
            logger.info("Compiling model with torch.compile()")
            self.model = torch.compile(self.model)

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return self._device

    @classmethod
    def from_pretrained(
        cls,
        config: ModelConfig,
        device: torch.device | str | None = None,
    ) -> FlamingoModel:
        """Load model from HuggingFace Hub or local path.

        Args:
            config: Model configuration
            device: Target device (if None, uses device_map from config)

        Returns:
            Initialized FlamingoModel
        """
        from transformers import AutoProcessor

        # Import the specific model class
        # Note: This may need adjustment based on the actual HuggingFace model name
        try:
            from transformers import AudioFlamingo3ForConditionalGeneration
        except ImportError:
            # Fallback for older transformers versions
            from transformers import AutoModelForVision2Seq as AudioFlamingo3ForConditionalGeneration
            logger.warning(
                "AudioFlamingo3ForConditionalGeneration not found, "
                "using AutoModelForVision2Seq as fallback"
            )

        logger.info(f"Loading model: {config.name}")
        logger.info(f"Dtype: {config.dtype}, Device map: {config.device_map}")

        # Prepare model kwargs
        model_kwargs = {
            "torch_dtype": config.torch_dtype,
            "trust_remote_code": config.trust_remote_code,
        }

        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            model_kwargs["device_map"] = None
        else:
            model_kwargs["device_map"] = config.device_map

        if config.max_memory:
            model_kwargs["max_memory"] = config.max_memory

        if config.revision:
            model_kwargs["revision"] = config.revision

        # Load processor
        processor = AutoProcessor.from_pretrained(
            config.name,
            trust_remote_code=config.trust_remote_code,
            revision=config.revision,
        )

        # Load model
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            config.name,
            **model_kwargs,
        )

        if device is not None:
            model = model.to(device)

        model.eval()

        return cls(model=model, processor=processor, config=config, device=device)

    def generate(
        self,
        audio: torch.Tensor,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        return_logprobs: bool = False,
    ) -> GenerationOutput:
        """Generate text response for audio input.

        Args:
            audio: Audio tensor, shape (samples,) or (channels, samples)
            prompt: Text prompt/instruction
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling
            top_k: Top-k sampling
            do_sample: Whether to use sampling
            num_beams: Number of beams for beam search
            repetition_penalty: Repetition penalty
            return_logprobs: Whether to return log probabilities

        Returns:
            GenerationOutput with generated text
        """
        # Ensure audio is 1D
        if audio.dim() == 2:
            audio = audio.mean(dim=0)

        # Build conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio.cpu().numpy()},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process inputs
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        # Generate
        with torch.inference_mode():
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature if do_sample else 1.0,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": do_sample,
                "num_beams": num_beams,
                "repetition_penalty": repetition_penalty,
                "pad_token_id": self.processor.tokenizer.pad_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
            }

            if return_logprobs:
                generation_kwargs["output_scores"] = True
                generation_kwargs["return_dict_in_generate"] = True
                outputs = self.model.generate(**inputs, **generation_kwargs)
                sequences = outputs.sequences
                # Compute log probabilities
                logprobs = self._compute_logprobs(outputs.scores, sequences)
            else:
                sequences = self.model.generate(**inputs, **generation_kwargs)
                logprobs = None

        # Decode output
        # Skip the input tokens
        input_len = inputs["input_ids"].shape[1]
        generated_ids = sequences[:, input_len:]
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Determine finish reason
        finish_reason = "stop"
        if generated_ids.shape[1] >= max_tokens:
            finish_reason = "length"

        return GenerationOutput(
            text=text.strip(),
            tokens=generated_ids[0].tolist(),
            logprobs=logprobs,
            finish_reason=finish_reason,
        )

    def _compute_logprobs(
        self,
        scores: tuple[torch.Tensor, ...],
        sequences: torch.Tensor,
    ) -> list[float]:
        """Compute log probabilities from generation scores."""
        logprobs = []
        for i, score in enumerate(scores):
            # Get log softmax
            log_probs = torch.log_softmax(score, dim=-1)
            # Get the token that was generated
            token_id = sequences[0, i + 1]  # +1 because scores don't include input
            logprobs.append(log_probs[0, token_id].item())
        return logprobs

    def extract_embedding(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,
    ) -> EmbeddingOutput:
        """Extract audio embeddings from the AF-Whisper encoder.

        Args:
            audio: Audio tensor, shape (samples,) or (channels, samples)
            sample_rate: Audio sample rate

        Returns:
            EmbeddingOutput with embeddings and timestamps
        """
        # Ensure audio is 1D
        if audio.dim() == 2:
            audio = audio.mean(dim=0)

        # Calculate duration
        audio_duration = len(audio) / sample_rate

        # Process audio through the audio encoder
        # This accesses the internal audio encoder of the model
        with torch.inference_mode():
            # Prepare audio input
            audio_np = audio.cpu().numpy()

            # Use processor to prepare audio features
            # Note: The exact method depends on the model's processor implementation
            audio_features = self.processor.feature_extractor(
                audio_np,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )

            # Move to device
            audio_features = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in audio_features.items()
            }

            # Extract embeddings from audio encoder
            # Access the audio encoder directly
            if hasattr(self.model, "audio_encoder"):
                encoder_outputs = self.model.audio_encoder(**audio_features)
                embeddings = encoder_outputs.last_hidden_state
            elif hasattr(self.model, "model") and hasattr(self.model.model, "audio_encoder"):
                encoder_outputs = self.model.model.audio_encoder(**audio_features)
                embeddings = encoder_outputs.last_hidden_state
            else:
                # Fallback: run through model and extract features
                raise NotImplementedError(
                    "Cannot access audio encoder directly. "
                    "Model architecture may differ from expected."
                )

            # embeddings shape: (batch, num_frames, embedding_dim)
            embeddings = embeddings.squeeze(0)  # (num_frames, embedding_dim)

        # Calculate frame timestamps
        num_frames = embeddings.shape[0]
        frame_timestamps = torch.linspace(0, audio_duration, num_frames)

        return EmbeddingOutput(
            embedding=embeddings.cpu(),
            frame_timestamps=frame_timestamps,
            audio_duration=audio_duration,
        )

    def get_memory_usage(self) -> dict[str, int]:
        """Get GPU memory usage in bytes."""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0}

        device_idx = self.device.index if self.device.type == "cuda" else 0
        return {
            "allocated": torch.cuda.memory_allocated(device_idx),
            "reserved": torch.cuda.memory_reserved(device_idx),
        }

    def __repr__(self) -> str:
        return (
            f"FlamingoModel(name={self.config.name!r}, "
            f"dtype={self.config.dtype}, device={self.device})"
        )
