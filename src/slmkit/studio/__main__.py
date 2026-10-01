"""`python -m slmkit.studio --port N`: the studio server itself (what `slm studio start` launches)."""

from __future__ import annotations

import argparse

import uvicorn

from slmkit.studio.app import create_app
from slmkit.studio.process import DEFAULT_PORT


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m slmkit.studio")
    parser.add_argument("--host", default="127.0.0.1", help="keep it local; see DESIGN 6.9")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--device", default="auto", help="where playground models run: auto, cuda, cpu"
    )
    args = parser.parse_args()
    uvicorn.run(create_app(device=args.device), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
