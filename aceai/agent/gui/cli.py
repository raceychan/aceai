"""Console entry point for the optional AceAI GUI server."""

import argparse
import importlib

from aceai.agent.config import load_config

from .server import AceAIGuiRuntime, build_gui_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AceAI GUI server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config()
    if config is None:
        raise SystemExit("AceAI config not found. Run `aceai` setup first.")
    try:
        uvicorn = importlib.import_module("uvicorn")
    except ModuleNotFoundError as exc:
        if exc.name == "uvicorn":
            raise SystemExit(
                "AceAI GUI server requires the optional gui dependencies. "
                "Install with `aceai[gui]`."
            ) from None
        raise
    app = build_gui_app(AceAIGuiRuntime(config=config))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
