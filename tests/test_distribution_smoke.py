import subprocess
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_base_wheel_install_does_not_require_provider_sdks(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path)
    python = _install_wheel(tmp_path / "base-venv", wheel)

    script = """
        import asyncio
        import importlib.util

        import aceai
        import aceai.llm
        from aceai import Agent, Executor, Graph, tool
        from aceai.llm import LLMMessage, LLMService
        from aceai.llm.models import LLMResponse, LLMStreamEvent

        assert importlib.util.find_spec("openai") is None
        assert importlib.util.find_spec("websockets") is None

        class FakeLLMService:
            async def stream(self, **request):
                yield LLMStreamEvent(
                    event_type="response.completed",
                    response=LLMResponse(text="base-ok"),
                )

        @tool
        def ping() -> str:
            return "pong"

        async def main():
            agent = Agent(
                prompt="Smoke test agent.",
                default_model="fake-model",
                llm_service=FakeLLMService(),
                executor=Executor(Graph(), [ping]),
                max_steps=1,
            )
            answer = await agent.ask("say smoke")
            assert answer == "base-ok"

        asyncio.run(main())

        try:
            import aceai.llm.openai
        except RuntimeError as exc:
            assert "aceai[openai]" in str(exc)
        else:
            raise AssertionError("OpenAI provider imported without the openai extra")

        print(aceai.__version__, Agent.__name__, Executor.__name__, LLMService.__name__, LLMMessage.__name__)
    """
    _run_python(python, script, cwd=tmp_path)


def test_provider_extras_make_provider_modules_importable(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path)
    python = _install_wheel(tmp_path / "providers-venv", f"{wheel}[providers]")

    script = """
        from aceai.llm.anthropic import Anthropic
        from aceai.llm.deepseek import DeepSeek
        from aceai.llm.openai import OpenAI
        from aceai.llm.openai_codex import OpenAICodex

        assert Anthropic.__name__ == "Anthropic"
        assert DeepSeek.__name__ == "DeepSeek"
        assert OpenAI.__name__ == "OpenAI"
        assert OpenAICodex.__name__ == "OpenAICodex"
    """
    _run_python(python, script, cwd=tmp_path)


def _build_wheel(tmp_path: Path) -> Path:
    dist_dir = tmp_path / "dist"
    _run(["uv", "build", "--wheel", "--out-dir", str(dist_dir)], cwd=PROJECT_ROOT)
    wheels = sorted(dist_dir.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def _install_wheel(venv_dir: Path, wheel: Path | str) -> Path:
    _run(["uv", "venv", str(venv_dir)])
    python = venv_dir / "bin" / "python"
    _run(["uv", "pip", "install", "--python", str(python), str(wheel)])
    return python


def _run_python(python: Path, script: str, *, cwd: Path) -> None:
    _run([str(python), "-c", textwrap.dedent(script)], cwd=cwd)


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
