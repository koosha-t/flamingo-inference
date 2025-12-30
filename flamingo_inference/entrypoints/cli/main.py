"""Main CLI entry point for Music Flamingo server."""

from __future__ import annotations

import sys


def main() -> int:
    """Main entry point.

    Returns:
        Exit code
    """
    # Simple subcommand dispatch without re-parsing serve arguments
    if len(sys.argv) < 2:
        _print_help()
        return 0

    command = sys.argv[1]

    if command in ("-h", "--help"):
        _print_help()
        return 0

    if command == "serve":
        from flamingo_inference.entrypoints.cli.serve import main as serve_main

        # Pass all args after "serve" to serve.py's parser
        sys.argv = ["flamingo"] + sys.argv[2:]
        return serve_main()

    elif command == "version":
        from flamingo_inference import __version__

        print(f"flamingo-inference {__version__}")
        return 0

    else:
        print(f"Unknown command: {command}")
        _print_help()
        return 1


def _print_help() -> None:
    """Print help message."""
    print(
        """Music Flamingo Inference Server

Usage: flamingo <command> [options]

Commands:
  serve     Start the inference server
  version   Show version information

Run 'flamingo serve --help' for server options."""
    )


if __name__ == "__main__":
    sys.exit(main())
