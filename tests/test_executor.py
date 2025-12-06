import pytest
from ididi import use

from aceai.executor import LoggingToolExecutor, ToolExecutor
from aceai.llm.interface import LLMToolCall
from aceai.tools import tool
from aceai.tools._param import Annotated, spec


class FakeGraph:
    def __init__(self):
        self.added_nodes: list[object] = []
        self.dep_results: dict[object, object] = {}
        self.resolve_calls: list[tuple[object, dict[str, object]]] = []

    def add_nodes(self, *nodes: object) -> None:
        self.added_nodes.extend(nodes)

    async def aresolve(self, node: object, **params: object) -> object:
        self.resolve_calls.append((node, params))
        return self.dep_results[node]

    def dependency_for(self, dependent_type: type) -> object:
        for node in self.added_nodes:
            if getattr(node, "dependent", None) is dependent_type:
                return node
        raise AssertionError(f"No dependency registered for {dependent_type}")


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
    return UserRepo("unused")


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


def build_tool(func):
    return tool(func)


@pytest.mark.anyio
async def test_tool_executor_executes_tool_with_dep_graph() -> None:
    graph = FakeGraph()
    describe_tool = build_tool(describe_user)
    executor = ToolExecutor(graph, [describe_tool])

    user_repo_node = graph.dependency_for(UserRepo)
    graph.dep_results[user_repo_node] = UserRepo("primary")

    call = LLMToolCall(
        name=describe_tool.name,
        arguments='{"user_id":7}',
        call_id="call-123",
    )

    encoded_result = await executor.execute_tool(call)

    assert encoded_result == '"primary:7"'
    assert graph.resolve_calls == [
        (user_repo_node, {"user_id": 7}),
    ]


@pytest.mark.anyio
async def test_tool_executor_awaits_async_tool_results() -> None:
    graph = FakeGraph()
    increment_tool = build_tool(async_increment)
    executor = ToolExecutor(graph, [increment_tool])

    call = LLMToolCall(
        name=increment_tool.name,
        arguments='{"value":2}',
        call_id="call-async",
    )

    result = await executor.execute_tool(call)

    assert result == "3"
    assert graph.resolve_calls == []


@pytest.mark.anyio
async def test_logging_tool_executor_logs_successful_calls() -> None:
    graph = FakeGraph()
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
async def test_logging_tool_executor_logs_and_reraises_failures() -> None:
    graph = FakeGraph()
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
    assert logger.exception_messages == [
        "Tool unreliable_tool failed after 1.25s"
    ]
    assert logger.success_messages == []
