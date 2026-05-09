import json

from aceai.agent.memory.subagent_artifacts import SubagentArtifactStore
from aceai.core.models import ToolExecutionResult
from aceai.llm.models import LLMToolCall


def test_subagent_artifact_store_archives_full_result_under_session(tmp_path) -> None:
    call = LLMToolCall(
        name="delegate_to_subagent",
        arguments='{"task":"inspect"}',
        call_id="call-1",
    )
    output = {
        "agent_id": "child-1",
        "run_id": "child-run-1",
        "status": "completed",
        "final_answer": "full child answer",
        "summary": "short child summary",
        "important_evidence": ["large evidence"],
        "tool_results": [
            {
                "tool_name": "read_text_file",
                "call_id": "call-read",
                "arguments": '{"path":"README.md"}',
                "output": "large file body",
                "error": None,
            }
        ],
        "step_count": 2,
    }
    model_output = {
        "type": "subagent_handoff",
        "agent_id": "child-1",
        "run_id": "child-run-1",
        "status": "completed",
        "task": "inspect",
        "handoff": "short child summary",
        "artifact_id": "child-1",
        "evidence": ["large evidence"],
        "step_count": 2,
        "tool_result_count": 1,
        "tool_names": ["read_text_file"],
    }
    result = ToolExecutionResult(
        call=call,
        output=json.dumps(output),
        model_output=json.dumps(model_output),
    )
    store = SubagentArtifactStore(tmp_path)

    archived = store.archive_tool_result(
        session_id="session-1",
        parent_run_id="parent-run-1",
        tool_result=result,
    )

    artifact_dir = tmp_path / "session-1" / "artifacts" / "parent-run-1" / "child-1"
    assert (artifact_dir / "manifest.json").exists()
    assert (artifact_dir / "handoff.json").exists()
    assert (artifact_dir / "final_answer.md").read_text(encoding="utf-8") == (
        "full child answer"
    )
    audit = json.loads(archived.output)
    handoff = json.loads(archived.model_output)
    assert audit["type"] == "subagent_audit"
    assert audit["manifest_path"] == (
        "session-1/artifacts/parent-run-1/child-1/manifest.json"
    )
    assert handoff["type"] == "subagent_handoff"
    assert handoff["artifact_id"] == audit["artifact_id"]
