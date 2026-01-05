FROM python:3.10-slim

WORKDIR /app

# System dependencies (git needed for transformers from git)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (for layer caching)
COPY pyproject.toml uv.lock ./

# Copy source code (needed for editable install)
COPY flamingo_inference/ ./flamingo_inference/
COPY configs/ ./configs/

# Install dependencies using uv with locked versions
RUN uv pip install --system --frozen -e .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -f http://localhost:8001/health || exit 1

EXPOSE 8001

# Use the CLI entrypoint
CMD ["flamingo", "serve", "--config", "configs/default.yaml", "--port", "8001"]
