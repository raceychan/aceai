from typing import AsyncGenerator

import httpx
import pytest
from ididi import Graph, use

from aceai.executor import LoggingToolExecutor, ToolExecutor
from aceai.llm.models import LLMToolCall
from aceai.tools import tool
from aceai.tools._param import Annotated, spec


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
    raise RuntimeError("expected failure")


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
    executor = ToolExecutor(graph, [describe_tool])

    call = LLMToolCall(
        name=describe_tool.name,
        arguments='{"user_id":7}',
        call_id="call-123",
    )

    encoded_result = await executor.execute_tool(call)

    assert encoded_result == '"primary:7"'


@pytest.mark.anyio
async def test_tool_executor_awaits_async_tool_results(graph: Graph) -> None:
    increment_tool = build_tool(async_increment)
    executor = ToolExecutor(graph, [increment_tool])

    call = LLMToolCall(
        name=increment_tool.name,
        arguments='{"value":2}',
        call_id="call-async",
    )

    result = await executor.execute_tool(call)

    assert result == "3"


@pytest.mark.anyio
async def test_logging_tool_executor_logs_successful_calls(graph: Graph) -> None:
    increment_tool = build_tool(async_increment)
    logger = FakeLogger()
    timer = StepTimer(10.0, 10.5)
    executor = LoggingToolExecutor(
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

    result = await executor.execute_tool(call)

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
    executor = LoggingToolExecutor(
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

    with pytest.raises(RuntimeError, match="expected failure"):
        await executor.execute_tool(call)

    assert logger.info_messages == [
        'Tool unreliable_tool starting (call_id=req-fail) with {"value":1}'
    ]
    assert logger.exception_messages == ["Tool unreliable_tool failed after 1.25s"]
    assert logger.success_messages == []


@pytest.mark.anyio
async def test_tool_executor_resolves_httpx_async_client(graph: Graph) -> None:
    client_tool = build_tool(identify_httpx_client)
    executor = ToolExecutor(graph, [client_tool])

    call = LLMToolCall(name=client_tool.name, arguments="{}", call_id="req-httpx")

    result = await executor.execute_tool(call)

    assert result == '"AsyncClient"'
