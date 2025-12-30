"""FastAPI server for Music Flamingo inference."""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

import aiohttp
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from flamingo_inference.config import FlamingoEngineConfig, ServerConfig
from flamingo_inference.engine import AsyncFlamingoEngine
from flamingo_inference.entrypoints.openai.protocol import (
    AnalyzeRequest,
    AnalyzeResponse,
    AudioData,
    AudioUrl,
    BatchJobRequest,
    BatchJobResponse,
    BatchJobStatus,
    CaptionRequest,
    CaptionResponse,
    CaptionUsage,
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatUsage,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    ErrorDetail,
    ErrorResponse,
    ModelInfo,
    ModelListResponse,
)

logger = logging.getLogger(__name__)

# Global engine instance
_engine: AsyncFlamingoEngine | None = None
_engine_config: FlamingoEngineConfig | None = None
_server_config: ServerConfig | None = None


async def get_engine() -> AsyncFlamingoEngine:
    """Get the engine instance."""
    global _engine
    if _engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized",
        )
    return _engine


async def get_audio_bytes(audio: AudioData | AudioUrl) -> bytes:
    """Get audio bytes from audio data or URL."""
    if isinstance(audio, AudioData):
        return base64.b64decode(audio.data)
    elif isinstance(audio, AudioUrl):
        async with aiohttp.ClientSession() as session:
            async with session.get(audio.url) as resp:
                if resp.status != 200:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to fetch audio from URL: {resp.status}",
                    )
                return await resp.read()
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid audio format",
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan management."""
    global _engine, _engine_config

    # Start engine
    if _engine_config is not None:
        _engine = AsyncFlamingoEngine(_engine_config)
        await _engine.start()
        logger.info("Engine started")

    yield

    # Stop engine
    if _engine is not None:
        await _engine.stop()
        _engine = None
        logger.info("Engine stopped")


def create_app(
    engine_config: FlamingoEngineConfig | None = None,
    server_config: ServerConfig | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        engine_config: Engine configuration
        server_config: Server configuration

    Returns:
        Configured FastAPI application
    """
    global _engine_config, _server_config

    _engine_config = engine_config or FlamingoEngineConfig()
    _server_config = server_config or ServerConfig()

    app = FastAPI(
        title="Music Flamingo Inference Server",
        description="OpenAI-compatible API for Music Flamingo audio-language model",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    if _server_config.cors.enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_server_config.cors.allow_origins,
            allow_credentials=_server_config.cors.allow_credentials,
            allow_methods=_server_config.cors.allow_methods,
            allow_headers=_server_config.cors.allow_headers,
        )

    # Error handler
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=ErrorDetail(
                    message=str(exc.detail),
                    type="invalid_request_error",
                )
            ).model_dump(),
        )

    # ========================================================================
    # Health Endpoints
    # ========================================================================

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        engine = await get_engine()
        return {
            "status": "healthy" if engine.is_ready else "unhealthy",
            "ready": engine.is_ready,
        }

    @app.get("/health/ready")
    async def readiness():
        """Readiness probe."""
        engine = await get_engine()
        if not engine.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Engine not ready",
            )
        return {"status": "ready"}

    @app.get("/health/live")
    async def liveness():
        """Liveness probe."""
        return {"status": "alive"}

    # ========================================================================
    # Model Endpoints
    # ========================================================================

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models():
        """List available models."""
        return ModelListResponse(
            data=[
                ModelInfo(
                    id="nvidia/music-flamingo-hf",
                    owned_by="nvidia",
                )
            ]
        )

    @app.get("/v1/models/{model_id}")
    async def get_model(model_id: str):
        """Get model information."""
        if model_id != "nvidia/music-flamingo-hf":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found: {model_id}",
            )
        return ModelInfo(id=model_id, owned_by="nvidia")

    # ========================================================================
    # Caption Endpoint
    # ========================================================================

    @app.post("/v1/audio/captions", response_model=CaptionResponse)
    async def create_caption(request: CaptionRequest):
        """Generate a caption for audio."""
        engine = await get_engine()
        request_id = f"cap-{uuid4().hex[:12]}"

        try:
            # Get audio bytes
            audio_bytes = await get_audio_bytes(request.audio)

            # Generate caption
            result = await engine.generate(
                audio=audio_bytes,
                prompt=request.prompt or _get_caption_prompt(request.style),
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )

            if not result.success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result.error or "Caption generation failed",
                )

            return CaptionResponse(
                id=request_id,
                model=request.model,
                caption=result.text or "",
                finish_reason=result.finish_reason or "stop",
                usage=CaptionUsage(
                    completion_tokens=len(result.tokens or []),
                    total_tokens=len(result.tokens or []),
                    audio_duration_seconds=result.audio_duration,
                ),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Caption generation error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    # ========================================================================
    # Embedding Endpoint
    # ========================================================================

    @app.post("/v1/audio/embeddings", response_model=EmbeddingResponse)
    async def create_embedding(request: EmbeddingRequest):
        """Extract audio embeddings."""
        engine = await get_engine()
        request_id = f"emb-{uuid4().hex[:12]}"

        try:
            # Get audio bytes
            audio_bytes = await get_audio_bytes(request.audio)

            # Extract embedding
            result = await engine.embed(audio=audio_bytes)

            if not result.success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result.error or "Embedding extraction failed",
                )

            # Format embedding
            if result.embedding is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No embedding returned",
                )

            embedding_data: list[list[float]] | str
            if request.encoding_format == "base64":
                embedding_data = base64.b64encode(
                    result.embedding.numpy().tobytes()
                ).decode()
            else:
                embedding_data = result.embedding.tolist()

            timestamps = None
            if result.frame_timestamps is not None:
                timestamps = result.frame_timestamps.tolist()

            return EmbeddingResponse(
                id=request_id,
                model=request.model,
                data=[
                    EmbeddingData(
                        embedding=embedding_data,
                        frame_timestamps=timestamps,
                    )
                ],
                usage=EmbeddingUsage(
                    audio_duration_seconds=result.audio_duration,
                    num_frames=result.embedding.shape[0],
                ),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Embedding extraction error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    # ========================================================================
    # Chat Completions Endpoint
    # ========================================================================

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(request: ChatCompletionRequest):
        """Interactive Q&A with audio."""
        engine = await get_engine()
        request_id = f"chatcmpl-{uuid4().hex[:12]}"

        try:
            # Extract audio and text from messages
            audio_bytes = None
            text_content = []

            for message in request.messages:
                if message.role == "user":
                    if isinstance(message.content, str):
                        text_content.append(message.content)
                    elif isinstance(message.content, list):
                        for item in message.content:
                            if hasattr(item, "text"):
                                text_content.append(item.text)
                            elif hasattr(item, "audio"):
                                audio_bytes = await get_audio_bytes(item.audio)

            if audio_bytes is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No audio found in messages",
                )

            prompt = " ".join(text_content) or "Describe this audio."

            # Generate response
            result = await engine.generate(
                audio=audio_bytes,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
            )

            if not result.success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result.error or "Generation failed",
                )

            return ChatCompletionResponse(
                id=request_id,
                model=request.model,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=result.text or "",
                        ),
                        finish_reason=result.finish_reason or "stop",
                    )
                ],
                usage=ChatUsage(
                    completion_tokens=len(result.tokens or []),
                    total_tokens=len(result.tokens or []),
                    audio_duration_seconds=result.audio_duration,
                ),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Chat completion error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    # ========================================================================
    # Analyze Endpoint
    # ========================================================================

    @app.post("/v1/audio/analyze", response_model=AnalyzeResponse)
    async def analyze_audio(request: AnalyzeRequest):
        """Generate caption and embeddings together."""
        engine = await get_engine()
        request_id = f"ana-{uuid4().hex[:12]}"

        try:
            # Get audio bytes
            audio_bytes = await get_audio_bytes(request.audio)

            # Analyze
            result = await engine.analyze(
                audio=audio_bytes,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            if not result.success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result.error or "Analysis failed",
                )

            # Format response
            embedding_data = None
            if request.include_embeddings and result.embedding is not None:
                if request.embedding_format == "base64":
                    emb = base64.b64encode(result.embedding.numpy().tobytes()).decode()
                else:
                    emb = result.embedding.tolist()

                timestamps = None
                if result.frame_timestamps is not None:
                    timestamps = result.frame_timestamps.tolist()

                embedding_data = EmbeddingData(
                    embedding=emb,
                    frame_timestamps=timestamps,
                )

            return AnalyzeResponse(
                id=request_id,
                model=request.model,
                caption=result.text or "",
                finish_reason=result.finish_reason or "stop",
                embedding=embedding_data,
                usage=CaptionUsage(
                    completion_tokens=len(result.tokens or []),
                    total_tokens=len(result.tokens or []),
                    audio_duration_seconds=result.audio_duration,
                ),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Analysis error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )

    # ========================================================================
    # Stats Endpoint
    # ========================================================================

    @app.get("/stats")
    async def get_stats():
        """Get engine statistics."""
        engine = await get_engine()
        return engine.get_stats()

    # ========================================================================
    # UI Endpoints
    # ========================================================================

    @app.get("/")
    async def redirect_to_ui():
        """Redirect root to UI."""
        return RedirectResponse(url="/ui/")

    # Mount static files for UI
    ui_dir = Path(__file__).parent / "ui"
    if ui_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    return app


def _get_caption_prompt(style: str) -> str:
    """Get caption prompt for style."""
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
    return prompts.get(style, prompts["detailed"])


def run_server(
    engine_config: FlamingoEngineConfig | None = None,
    server_config: ServerConfig | None = None,
) -> None:
    """Run the server.

    Args:
        engine_config: Engine configuration
        server_config: Server configuration
    """
    import uvicorn

    server_config = server_config or ServerConfig()
    app = create_app(engine_config, server_config)

    uvicorn.run(
        app,
        host=server_config.host,
        port=server_config.port,
        workers=server_config.workers,
        timeout_keep_alive=server_config.timeout_keep_alive,
        access_log=server_config.logging.access_log,
    )
