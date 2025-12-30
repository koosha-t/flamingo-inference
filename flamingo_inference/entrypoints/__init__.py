"""Entrypoints for Music Flamingo server."""

from flamingo_inference.entrypoints.api_server import create_app, run_server

__all__ = [
    "create_app",
    "run_server",
]
