"""Console entry point for the optional AceAI GUI server."""

import argparse
import importlib
from pathlib import Path

from aceai.agent.config import AgentAppConfig, load_config

from .server import AceAIGuiRuntime, build_gui_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AceAI GUI server.")
    parser.add_argument("command", nargs="?", choices=("serve", "schema"), default="serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--out", type=Path, help="Write the generated OpenAPI schema.")
    return parser


def _load_runtime() -> AceAIGuiRuntime:
    config = load_config()
    if config is None:
        raise SystemExit("AceAI config not found. Run `aceai` setup first.")
    return AceAIGuiRuntime(config=config)


def _schema_runtime() -> AceAIGuiRuntime:
    return AceAIGuiRuntime(
        config=AgentAppConfig(
            provider="openai",
            api_key="schema",
            model="gpt-5.5",
            default_model="gpt-5.5",
        )
    )


def _write_schema(path: Path | None) -> None:
    try:
        from lihil.utils.json import encoder_factory
    except ModuleNotFoundError as exc:
        if exc.name == "lihil":
            raise SystemExit(
                "AceAI GUI schema generation requires the optional gui dependencies. "
                "Install with `aceai[gui]`."
            ) from None
        raise
    app = build_gui_app(_schema_runtime())
    schema = app.genereate_oas()
    data = encoder_factory()(schema)
    if path is None:
        print(data.decode())
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "schema":
        _write_schema(args.out)
        return
    try:
        uvicorn = importlib.import_module("uvicorn")
    except ModuleNotFoundError as exc:
        if exc.name == "uvicorn":
            raise SystemExit(
                "AceAI GUI server requires the optional gui dependencies. "
                "Install with `aceai[gui]`."
            ) from None
        raise
    app = build_gui_app(_load_runtime())
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
