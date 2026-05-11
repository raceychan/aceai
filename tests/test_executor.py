from pathlib import Path
from typing import AsyncGenerator

import httpx
import pytest
from ididi import Graph, use
from msgspec import Struct

from aceai.llm.errors import AceAIRuntimeError
from aceai.core.executor import (
    DummyExecutor,
    Executor,
    LoggingExecutor,
    ToolExecutionError,
)
from aceai.core.output_truncation import DEFAULT_TRUNCATED_OUTPUT_TOKEN_BUDGET
from aceai.core.run_state import ToolRunState
from aceai.llm.models import LLMToolCall
from aceai.core.tools import tool
from aceai.core.tools._tool_sig import Annotated, spec


class FakeLogger:
    def __init__(self):
        self.info_messages: list[str] = []
        self.success_messages: list[str] = []
        self.exception_messages: list[str] = []

    def info(self, msg: str, /, **_: object) -> None:
        self.info_messages.append(msg)

    def success(self, msg: str, /, **_: object) -> None:
        self.success_messages.append(msg)

    def exception(self, msg: str, /, **_: object) -> None:
        self.exception_messages.append(msg)


class StepTimer:
    def __init__(self, *ticks: float):
        self._ticks = iter(ticks)

    def __call__(self) -> float:
        return next(self._ticks)


class UserRepo:
    def __init__(self, label: str):
        self.label = label


def provide_user_repo() -> UserRepo:
    return UserRepo("primary")


def describe_user(
    repo: Annotated[UserRepo, use(provide_user_repo)],
    user_id: Annotated[int, spec(description="User identifier")],
) -> str:
    return f"{repo.label}:{user_id}"


async def async_increment(
    value: Annotated[int, spec(description="Value to increment")],
) -> int:
    return value + 1


def unreliable_tool(
    value: Annotated[int, spec(description="Value that triggers failure")],
) -> int:
    raise AceAIRuntimeError("expected failure")


def echo_message(
    message: Annotated[str, spec(description="Message to echo")],
) -> str:
    return message


def long_message() -> str:
    return "tool-output-line\n" * 5000


def build_async_client() -> httpx.AsyncClient:
    return httpx.AsyncClient()


async def identify_httpx_client(
    client: Annotated[httpx.AsyncClient, use(build_async_client)],
) -> str:
    try:
        return client.__class__.__name__
    finally:
        await client.aclose()


def build_tool(func):
    return tool(func)


def write_skill(root: Path, name: str, description: str, body: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: {description}",
                "---",
                body,
            ]
        ),
        encoding="utf-8",
    )
    return skill_dir


async def execute_call(
    executor: Executor,
    call: LLMToolCall,
    tool_state: ToolRunState | None = None,
) -> str:
    result = await executor.execute(
        executor.resolve_invocation(call),
        tool_state=tool_state or ToolRunState(),
    )
    assert result.truncated_output == result.output
    return result.output


@pytest.mark.anyio
async def test_tool_executor_truncates_default_truncated_output(graph: Graph) -> None:
    executor = Executor(graph, [tool(long_message)])
    call = LLMToolCall(
        name="long_message",
        arguments="{}",
        call_id="call-long-message",
    )

    result = await executor.execute(
        executor.resolve_invocation(call),
        tool_state=ToolRunState(),
    )

    assert result.output.startswith('"tool-output-line\\n')
    assert result.truncated_output != result.output
    assert "tokens truncated" in result.truncated_output
    assert len(result.truncated_output) < DEFAULT_TRUNCATED_OUTPUT_TOKEN_BUDGET * 5


def test_dummy_executor_has_no_tools() -> None:
    executor = DummyExecutor()

    assert executor.select_tools() == []
    assert executor.select_tools(include={"missing"}) == []
    assert executor.select_tools(exclude={"missing"}) == []


def test_dummy_executor_rejects_tool_invocation() -> None:
    executor = DummyExecutor()
    call = LLMToolCall(name="missing", arguments="{}", call_id="call-missing")

    with pytest.raises(KeyError, match="missing"):
        executor.resolve_invocation(call)


def test_executor_loads_skill_registry_and_prompt_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "project"
    write_skill(home / ".aceai" / "skills", "release", "Release workflow.", "# Release")
    write_skill(cwd / ".agents" / "skills", "review", "Review workflow.", "# Review")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)

    executor = Executor(Graph(), [], skill_path="auto")

    assert set(executor.skill_registry.skills) == {"release", "review"}
    assert "<available_skills>" in executor.prompt_instructions
    assert "<name>release</name>" in executor.prompt_instructions
    assert "<name>review</name>" in executor.prompt_instructions
    assert "skills_list" in executor.tools
    assert "skill_view" in executor.tools


def test_executor_loads_extra_skill_paths_after_user_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "project"
    extra = tmp_path / "builtin"
    write_skill(home / ".aceai" / "skills", "release", "Release workflow.", "# Release")
    write_skill(cwd / ".agents" / "skills", "review", "Review workflow.", "# Review")
    write_skill(extra, "skill-creator", "Create skills.", "# Skill Creator")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)

    executor = Executor(
        Graph(),
        [],
        skill_path="auto",
        extra_skill_paths=(extra,),
    )

    assert set(executor.skill_registry.skills) == {
        "release",
        "review",
        "skill-creator",
    }
    assert "<name>skill-creator</name>" in executor.prompt_instructions


def test_executor_extra_skill_paths_do_not_override_user_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "project"
    extra = tmp_path / "builtin"
    write_skill(
        home / ".aceai" / "skills",
        "skill-creator",
        "User skill creator.",
        "# User Skill Creator",
    )
    write_skill(extra, "skill-creator", "Builtin skill creator.", "# Builtin")
    monkeypatch.setenv("HOME", str(home))
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    executor = Executor(
        Graph(),
        [],
        skill_path="auto",
        extra_skill_paths=(extra,),
    )

    skill = executor.skill_registry.get("skill-creator")
    assert skill.description == "User skill creator."
    assert skill.read_instructions() == "# User Skill Creator"


def test_executor_disable_skill_path_skips_extra_skill_paths(tmp_path: Path) -> None:
    extra = tmp_path / "builtin"
    write_skill(extra, "skill-creator", "Create skills.", "# Skill Creator")

    executor = Executor(
        Graph(),
        [],
        skill_path="disable",
        extra_skill_paths=(extra,),
    )

    assert executor.skill_registry.skills == {}
    assert "skills_list" not in executor.tools


def test_executor_filters_enabled_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "project"
    write_skill(home / ".aceai" / "skills", "release", "Release workflow.", "# Release")
    write_skill(cwd / ".agents" / "skills", "review", "Review workflow.", "# Review")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)

    executor = Executor(
        Graph(),
        [],
        skill_path="auto",
        enabled_skill_names=("review",),
    )

    assert set(executor.skill_registry.skills) == {"review"}
    assert "<name>review</name>" in executor.prompt_instructions
    assert "<name>release</name>" not in executor.prompt_instructions


@pytest.fixture
async def graph() -> AsyncGenerator[Graph, None]:
    graph = Graph()
    try:
        yield graph
    finally:
        graph._workers.shutdown(wait=True)


@pytest.mark.anyio
async def test_tool_executor_executes_tool_with_dep_graph(graph: Graph) -> None:
    describe_tool = build_tool(describe_user)
    executor = Executor(graph, [describe_tool])

    call = LLMToolCall(
        name=describe_tool.name,
        arguments='{"user_id":7}',
        call_id="call-123",
    )

    encoded_result = await execute_call(executor, call)

    assert encoded_result == '"primary:7"'


@pytest.mark.anyio
async def test_tool_executor_awaits_async_tool_results(graph: Graph) -> None:
    increment_tool = build_tool(async_increment)
    executor = Executor(graph, [increment_tool])

    call = LLMToolCall(
        name=increment_tool.name,
        arguments='{"value":2}',
        call_id="call-async",
    )

    result = await execute_call(executor, call)

    assert result == "3"


@pytest.mark.anyio
async def test_tool_executor_returns_invalid_arguments_as_tool_execution_error(
    graph: Graph,
) -> None:
    message_tool = build_tool(echo_message)
    executor = Executor(graph, [message_tool])
    call = LLMToolCall(
        name=message_tool.name,
        arguments='{"query":"hello"}',
        call_id="call-invalid-args",
    )

    with pytest.raises(ToolExecutionError) as exc_info:
        await execute_call(executor, call)

    assert str(exc_info.value) == (
        "Invalid arguments for tool echo_message: Object missing required field `message`"
    )


@pytest.mark.anyio
async def test_logging_tool_executor_logs_successful_calls(graph: Graph) -> None:
    increment_tool = build_tool(async_increment)
    logger = FakeLogger()
    timer = StepTimer(10.0, 10.5)
    executor = LoggingExecutor(
        graph=graph,
        tools=[increment_tool],
        logger=logger,
        timer=timer,
    )

    call = LLMToolCall(
        name=increment_tool.name,
        arguments='{"value":2}',
        call_id="req-success",
    )

    result = await execute_call(executor, call)

    assert result == "3"
    assert logger.info_messages == [
        'Tool async_increment starting (call_id=req-success) with {"value":2}'
    ]
    assert logger.success_messages == [
        "Tool async_increment finished in 0.50s, result: 3"
    ]
    assert logger.exception_messages == []


@pytest.mark.anyio
async def test_logging_tool_executor_logs_and_reraises_failures(graph: Graph) -> None:
    failing_tool = build_tool(unreliable_tool)
    logger = FakeLogger()
    timer = StepTimer(5.0, 6.25)
    executor = LoggingExecutor(
        graph=graph,
        tools=[failing_tool],
        logger=logger,
        timer=timer,
    )

    call = LLMToolCall(
        name=failing_tool.name,
        arguments='{"value":1}',
        call_id="req-fail",
    )

    with pytest.raises(AceAIRuntimeError, match="expected failure"):
        await execute_call(executor, call)

    assert logger.info_messages == [
        'Tool unreliable_tool starting (call_id=req-fail) with {"value":1}'
    ]
    assert logger.exception_messages == ["Tool unreliable_tool failed after 1.25s"]
    assert logger.success_messages == []


@pytest.mark.anyio
async def test_tool_executor_resolves_httpx_async_client(graph: Graph) -> None:
    client_tool = build_tool(identify_httpx_client)
    executor = Executor(graph, [client_tool])

    call = LLMToolCall(name=client_tool.name, arguments="{}", call_id="req-httpx")

    result = await execute_call(executor, call)

    assert result == '"AsyncClient"'


class Payload(Struct):
    value: int


def build_struct(value: Annotated[int, spec(description="Value to wrap")]) -> Payload:
    return Payload(value)


@pytest.mark.anyio
async def test_tool_executor_encodes_struct_return(graph: Graph) -> None:
    struct_tool = build_tool(build_struct)
    executor = Executor(graph, [struct_tool])

    call = LLMToolCall(
        name=struct_tool.name,
        arguments='{"value":5}',
        call_id="req-struct",
    )

    encoded = await execute_call(executor, call)

    assert encoded == '{"value":5}'


@pytest.mark.anyio
async def test_tool_executor_exposes_tool_specs(graph: Graph) -> None:
    echo_tool = build_tool(echo_message)
    executor = Executor(graph, [echo_tool])

    first = executor.all_tools
    second = executor.all_tools

    assert first is second
    schema_names = {spec.name for spec in first}
    assert echo_tool.name in schema_names


@pytest.mark.anyio
async def test_tool_executor_enforces_max_calls_per_run(graph: Graph) -> None:
    executed: list[int] = []

    def tick() -> int:
        executed.append(1)
        return len(executed)

    tick_tool = tool(max_calls_per_run=1)(tick)
    executor = Executor(graph, [tick_tool])
    tool_state = ToolRunState()

    first = await execute_call(
        executor,
        LLMToolCall(name=tick_tool.name, arguments="{}", call_id="call-1"),
        tool_state,
    )
    second = await execute_call(
        executor,
        LLMToolCall(name=tick_tool.name, arguments="{}", call_id="call-2"),
        tool_state,
    )

    assert first == "1"
    assert second == (
        "the tool tick exceeds its max calls in this run, do not call it again"
    )
    assert executed == [1]


@pytest.mark.anyio
async def test_tool_executor_resolves_invocation_with_approval_metadata(
    graph: Graph,
) -> None:
    approved_tool = tool(require_approval=True)(echo_message)
    executor = Executor(graph, [approved_tool])
    call = LLMToolCall(
        name=approved_tool.name,
        arguments='{"message":"hello"}',
        call_id="call-approval",
    )

    invocation = executor.resolve_invocation(call)
    result = await executor.execute(invocation, tool_state=ToolRunState())

    assert invocation.call is call
    assert invocation.tool is approved_tool
    assert invocation.approval_required is True
    assert result.output == '"hello"'
    assert result.truncated_output == '"hello"'
