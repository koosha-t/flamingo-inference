# Music Flamingo Inference Server

High-availability inference server for the Music Flamingo audio-language model with an OpenAI-compatible REST API.

## Overview

This server provides production-ready inference for [NVIDIA's Music Flamingo](https://github.com/NVIDIA/audio-flamingo) model, enabling:

- **Caption Generation**: Rich text descriptions of audio content
- **Embedding Extraction**: 1536-dimensional AF-Whisper embeddings
- **Interactive Q&A**: Ask questions about audio content
- **Combined Analysis**: Caption + embeddings in a single request

Designed for both real-time inference and batch processing on A100/H100 GPUs.

## Features

- **OpenAI-Compatible API** - Drop-in replacement patterns for easy integration
- **Single & Multi-GPU Support** - Data parallelism for scaling across GPUs
- **Priority Scheduling** - Audio-aware request batching and prioritization
- **Browser UI** - Built-in testing interface at `/ui`
- **Prometheus Metrics** - Production observability at `/metrics`
- **Health Checks** - Kubernetes-ready liveness and readiness probes

## Installation

```bash
git clone https://github.com/koosha-t/flamingo-inference.git
cd flamingo-inference
pip install -e .
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU inference)
- 24GB+ GPU memory (A100/H100 recommended)

## Quick Start

### Start the Server

```bash
# Default configuration (single GPU)
python -m flamingo_inference.entrypoints.cli.serve

# With custom config
python -m flamingo_inference.entrypoints.cli.serve --config configs/production.yaml

# Specify host/port
python -m flamingo_inference.entrypoints.cli.serve --host 0.0.0.0 --port 8000
```

### Open the UI

Navigate to [http://localhost:8000/ui](http://localhost:8000/ui) for the browser-based testing interface.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/captions` | POST | Generate audio captions |
| `/v1/audio/embeddings` | POST | Extract AF-Whisper embeddings |
| `/v1/chat/completions` | POST | Interactive Q&A with audio |
| `/v1/audio/analyze` | POST | Combined caption + embeddings |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health status |
| `/health/ready` | GET | Readiness probe |
| `/health/live` | GET | Liveness probe |
| `/stats` | GET | Engine statistics |

### Example: Generate Caption

```bash
curl -X POST http://localhost:8000/v1/audio/captions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/music-flamingo-hf",
    "audio": {"url": "https://example.com/song.wav"},
    "style": "detailed"
  }'
```

### Example: Extract Embeddings

```bash
curl -X POST http://localhost:8000/v1/audio/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/music-flamingo-hf",
    "audio": {"data": "<base64-encoded-audio>"}
  }'
```

### Example: Chat Completion

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/music-flamingo-hf",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "audio", "audio": {"url": "https://example.com/song.wav"}},
          {"type": "text", "text": "What instruments are playing in this music?"}
        ]
      }
    ]
  }'
```

## Configuration

Configuration files are in `configs/`:

- `default.yaml` - Development settings (single GPU)
- `production.yaml` - Production settings (multi-GPU, optimizations)

### Key Configuration Options

```yaml
model:
  model_id: "nvidia/music-flamingo-hf"
  device: "cuda"
  dtype: "bfloat16"
  compile: true  # Enable torch.compile

audio:
  sample_rate: 16000
  max_duration: 600  # 10 minutes max

executor:
  num_workers: 1      # GPUs to use
  multiprocessing: false

scheduler:
  max_batch_size: 8
  max_waiting_time: 0.1
```

## Python Client

### Synchronous Client

```python
from sorna.enrichment import FlamingoClient

client = FlamingoClient(base_url="http://localhost:8000")

# Generate caption
result = client.caption("song.wav", style="detailed")
print(result.caption)

# Extract embeddings
result = client.embed("song.wav")
print(result.embedding.shape)  # (num_frames, 1536)

# Combined analysis
result = client.analyze("song.wav")
print(result.caption)
print(result.embedding.shape)
```

### Async Client (High Throughput)

```python
import asyncio
from sorna.enrichment import AsyncFlamingoClient

async def main():
    async with AsyncFlamingoClient(base_url="http://localhost:8000") as client:
        # Single request
        result = await client.analyze("song.wav")

        # Batch processing with concurrency
        audio_files = ["song1.wav", "song2.wav", "song3.wav"]
        async for path, result in client.batch_analyze(
            audio_files,
            output_dir="embeddings/",
            concurrency=16
        ):
            if not isinstance(result, Exception):
                print(f"{path}: {result.caption[:50]}...")

asyncio.run(main())
```

### Process Manifest

```python
from sorna.enrichment import process_manifest

# Enrich a JSONL manifest with Flamingo captions and embeddings
output_path = await process_manifest(
    manifest_path="dataset.jsonl",
    output_dir="enriched/",
    server_url="http://localhost:8000",
    concurrency=16
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Server                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ /caption │ │ /embed   │ │ /chat    │ │ /analyze │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
│       └────────────┴────────────┴────────────┘                 │
│                           │                                     │
│                  ┌────────▼────────┐                           │
│                  │ AsyncFlamingo   │                           │
│                  │    Engine       │                           │
│                  └────────┬────────┘                           │
│                           │                                     │
│                  ┌────────▼────────┐                           │
│                  │    Scheduler    │  Priority Queue           │
│                  │  (Audio-Aware)  │  + Batching               │
│                  └────────┬────────┘                           │
│                           │                                     │
│         ┌─────────────────┼─────────────────┐                  │
│         │                 │                 │                  │
│  ┌──────▼──────┐  ┌───────▼──────┐  ┌──────▼──────┐           │
│  │   Worker 0  │  │   Worker 1   │  │   Worker N  │           │
│  │   (GPU 0)   │  │   (GPU 1)    │  │   (GPU N)   │           │
│  └──────┬──────┘  └───────┬──────┘  └──────┬──────┘           │
│         │                 │                 │                  │
│  ┌──────▼──────┐  ┌───────▼──────┐  ┌──────▼──────┐           │
│  │  Flamingo   │  │   Flamingo   │  │  Flamingo   │           │
│  │   Model     │  │    Model     │  │   Model     │           │
│  └─────────────┘  └──────────────┘  └─────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## Model Details

| Property | Value |
|----------|-------|
| Model | `nvidia/music-flamingo-hf` |
| Parameters | ~8B |
| LLM Backbone | Qwen2.5-7B |
| Audio Encoder | AF-Whisper |
| Embedding Dim | 1536 |
| Max Audio | 10 minutes |
| Sample Rate | 16kHz |

## Caption Styles

| Style | Description |
|-------|-------------|
| `detailed` | Comprehensive description including genre, mood, instrumentation, tempo, melody, harmony, and production techniques |
| `brief` | Concise 2-3 sentence summary |
| `technical` | Technical analysis of time signature, key, chord progressions, and audio quality |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black flamingo_inference/
ruff check flamingo_inference/
```

## License

This inference server code is provided under the MIT License.

**Important**: The Music Flamingo model checkpoints are subject to the [NVIDIA OneWay Noncommercial License](https://github.com/NVIDIA/audio-flamingo) (non-commercial use only). The Qwen-2.5 backbone is subject to the [Qwen Research License Agreement](https://github.com/QwenLM/Qwen). Please review these licenses before use.

## Acknowledgments

- [NVIDIA Audio Flamingo](https://github.com/NVIDIA/audio-flamingo) - Original model
- [vLLM](https://github.com/vllm-project/vllm) - Architecture inspiration
