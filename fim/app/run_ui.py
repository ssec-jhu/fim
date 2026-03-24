"""CLI entry point for the FIM web UI (``fim-ui``)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fim-ui",
        description="Run the FIM web UI (FastAPI + static frontend).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: %(default)s). Use 0.0.0.0 to listen on all interfaces.",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port (default: %(default)s)")
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload (useful for production or scripted runs).",
    )
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "fim.app.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
    )


if __name__ == "__main__":
    main()
