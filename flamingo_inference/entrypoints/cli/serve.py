"""CLI command for running the server."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Setup logging configuration.

    Args:
        level: Logging level
        json_format: Use JSON format
    """
    import logging.config

    if json_format:
        try:
            import json

            class JsonFormatter(logging.Formatter):
                def format(self, record):
                    log_data = {
                        "timestamp": self.formatTime(record),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                    }
                    if record.exc_info:
                        log_data["exception"] = self.formatException(record.exc_info)
                    return json.dumps(log_data)

            handler = logging.StreamHandler()
            handler.setFormatter(JsonFormatter())
            logging.root.addHandler(handler)
            logging.root.setLevel(getattr(logging, level.upper()))
        except Exception:
            logging.basicConfig(level=level.upper())
    else:
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Music Flamingo Inference Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )

    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/music-flamingo-hf",
        help="Model name or path",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Model data type",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for model loading",
    )

    # Server options
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of server workers",
    )

    # Executor options
    parser.add_argument(
        "--executor-type",
        type=str,
        choices=["uniproc", "multiproc"],
        default="uniproc",
        help="Executor type",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs (e.g., '0,1,2')",
    )

    # Scheduler options
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Maximum batch size",
    )
    parser.add_argument(
        "--max-waiting-requests",
        type=int,
        default=1000,
        help="Maximum requests in queue",
    )

    # Logging options
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Use JSON log format",
    )

    # Feature flags
    parser.add_argument(
        "--enable-metrics",
        action="store_true",
        default=True,
        help="Enable Prometheus metrics",
    )
    parser.add_argument(
        "--disable-metrics",
        action="store_true",
        help="Disable Prometheus metrics",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile for faster inference",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the serve command.

    Returns:
        Exit code
    """
    args = parse_args()

    # Setup logging
    setup_logging(level=args.log_level, json_format=args.json_logs)

    logger.info("Starting Music Flamingo Inference Server")

    try:
        from flamingo_inference.config import (
            ExecutorType,
            FlamingoEngineConfig,
            ServerConfig,
        )
        from flamingo_inference.entrypoints.api_server import run_server

        # Load or create configuration
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            engine_config = FlamingoEngineConfig.from_yaml(args.config)
            server_config = ServerConfig.from_yaml(args.config)
        else:
            engine_config = FlamingoEngineConfig()
            server_config = ServerConfig()

        # Override with command line arguments
        engine_config.model.name = args.model
        engine_config.model.dtype = args.dtype
        engine_config.model.device_map = args.device_map
        engine_config.model.torch_compile = args.torch_compile

        engine_config.scheduler.max_batch_size = args.max_batch_size
        engine_config.scheduler.max_waiting_requests = args.max_waiting_requests

        engine_config.executor.type = ExecutorType(args.executor_type)
        if args.gpu_ids:
            engine_config.executor.gpu_ids = [
                int(x.strip()) for x in args.gpu_ids.split(",")
            ]

        server_config.host = args.host
        server_config.port = args.port
        server_config.workers = args.workers
        server_config.metrics.enabled = args.enable_metrics and not args.disable_metrics

        # Log configuration
        logger.info(f"Model: {engine_config.model.name}")
        logger.info(f"Dtype: {engine_config.model.dtype}")
        logger.info(f"Executor: {engine_config.executor.type.value}")
        logger.info(f"GPU IDs: {engine_config.executor.gpu_ids or 'auto'}")
        logger.info(f"Server: {server_config.host}:{server_config.port}")

        # Run server
        run_server(engine_config, server_config)
        return 0

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        return 0
    except Exception as e:
        logger.exception(f"Server error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
