from pathlib import Path

import pytest
from ididi import Graph

from aceai.agent.executor import RunState, ToolExecutor
from aceai.llm.models import LLMToolCall


@pytest.fixture
def executor() -> ToolExecutor:
    return ToolExecutor(Graph(), [])


@pytest.mark.anyio
async def test_builtin_read_text_file(tmp_path: Path, executor: ToolExecutor) -> None:
    file = tmp_path / "note.txt"
    file.write_text("hello", encoding="utf-8")

    result = await executor.execute_tool(
        LLMToolCall(
            name="read_text_file",
            arguments=f'{{"path":"{file}"}}',
            call_id="read-1",
        ),
        run_state=RunState(),
    )

    assert result == '"hello"'


@pytest.mark.anyio
async def test_builtin_list_dir(tmp_path: Path, executor: ToolExecutor) -> None:
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a").mkdir()

    result = await executor.execute_tool(
        LLMToolCall(
            name="list_dir",
            arguments=f'{{"path":"{tmp_path}"}}',
            call_id="list-1",
        ),
        run_state=RunState(),
    )

    assert '"name":"a"' in result
    assert '"kind":"directory"' in result
    assert '"name":"b.txt"' in result
    assert '"kind":"file"' in result


@pytest.mark.anyio
async def test_builtin_search_text(tmp_path: Path, executor: ToolExecutor) -> None:
    (tmp_path / "one.txt").write_text("alpha\nneedle\n", encoding="utf-8")
    (tmp_path / "two.txt").write_text("nope\n", encoding="utf-8")

    result = await executor.execute_tool(
        LLMToolCall(
            name="search_text",
            arguments=f'{{"query":"needle","path":"{tmp_path}"}}',
            call_id="search-1",
        ),
        run_state=RunState(),
    )

    assert '"line_number":2' in result
    assert '"line":"needle"' in result


@pytest.mark.anyio
async def test_builtin_edit_text_file_requires_unique_match(
    tmp_path: Path, executor: ToolExecutor
) -> None:
    file = tmp_path / "note.txt"
    file.write_text("one two one", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly once"):
        await executor.execute_tool(
            LLMToolCall(
                name="edit_text_file",
                arguments=f'{{"path":"{file}","old_string":"one","new_string":"three"}}',
                call_id="edit-1",
            ),
            run_state=RunState(),
        )


@pytest.mark.anyio
async def test_builtin_edit_text_file_replaces_once(
    tmp_path: Path, executor: ToolExecutor
) -> None:
    file = tmp_path / "note.txt"
    file.write_text("hello world", encoding="utf-8")

    result = await executor.execute_tool(
        LLMToolCall(
            name="edit_text_file",
            arguments=f'{{"path":"{file}","old_string":"world","new_string":"aceai"}}',
            call_id="edit-2",
        ),
        run_state=RunState(),
    )

    assert '"replacements":1' in result
    assert file.read_text(encoding="utf-8") == "hello aceai"


@pytest.mark.anyio
async def test_builtin_run_command(executor: ToolExecutor) -> None:
    result = await executor.execute_tool(
        LLMToolCall(
            name="run_command",
            arguments='{"command":["python","-c","print(123)"],"cwd":"."}',
            call_id="cmd-1",
        ),
        run_state=RunState(),
    )

    assert '"returncode":0' in result
    assert '"stdout":"123\\n"' in result
