"""Main CLI entry point for Music Flamingo server."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    """Main entry point.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Music Flamingo Inference Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the inference server",
    )
    serve_parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    serve_parser.add_argument(
        "--gpu-ids",
        type=str,
        help="Comma-separated GPU IDs",
    )

    # Version command
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command == "serve":
        from flamingo_inference.entrypoints.cli.serve import main as serve_main
        # Re-parse with full serve arguments
        sys.argv = ["flamingo", "serve"] + sys.argv[2:]
        return serve_main()
    elif args.command == "version":
        from flamingo_inference import __version__
        print(f"flamingo-inference {__version__}")
        return 0
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
