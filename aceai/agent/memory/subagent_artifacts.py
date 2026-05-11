import json
from pathlib import Path

from aceai.core.helpers.string import uuid_str
from aceai.core.models import ToolExecutionResult


class SubagentArtifactStore:
    """Archives delegated child-agent audit bodies under the owning session."""

    def __init__(self, sessions_root: Path) -> None:
        self.sessions_root = sessions_root

    def archive_tool_result(
        self,
        *,
        session_id: str,
        parent_run_id: str,
        tool_result: ToolExecutionResult,
    ) -> ToolExecutionResult:
        payload = json.loads(tool_result.output)
        handoff = json.loads(tool_result.truncated_output)
        agent_id = payload["agent_id"]
        artifact_id = handoff["artifact_id"]
        artifact_dir = self.sessions_root / session_id / "artifacts" / parent_run_id / agent_id
        tool_results_dir = artifact_dir / "tool-results"
        tool_results_dir.mkdir(parents=True, exist_ok=True)

        tool_artifacts: list[dict[str, object]] = []
        for child_tool_result in payload["tool_results"]:
            tool_artifact_id = uuid_str()
            tool_artifact_dir = tool_results_dir / tool_artifact_id
            tool_artifact_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "artifact_id": tool_artifact_id,
                "tool_name": child_tool_result["tool_name"],
                "tool_call_id": child_tool_result["call_id"],
                "has_error": child_tool_result["error"] is not None,
            }
            (tool_artifact_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (tool_artifact_dir / "arguments.json").write_text(
                child_tool_result["arguments"],
                encoding="utf-8",
            )
            (tool_artifact_dir / "output.txt").write_text(
                child_tool_result["output"],
                encoding="utf-8",
            )
            if child_tool_result["error"] is not None:
                (tool_artifact_dir / "error.txt").write_text(
                    child_tool_result["error"],
                    encoding="utf-8",
                )
            tool_artifacts.append(metadata)

        (artifact_dir / "final_answer.md").write_text(
            payload["final_answer"],
            encoding="utf-8",
        )
        (artifact_dir / "handoff.json").write_text(
            json.dumps(handoff, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest = {
            "type": "subagent_artifact_manifest",
            "artifact_id": artifact_id,
            "thread_id": payload["thread_id"],
            "agent_id": agent_id,
            "run_id": payload["run_id"],
            "parent_run_id": parent_run_id,
            "status": payload["status"],
            "summary": payload["summary"],
            "step_count": payload["step_count"],
            "tool_results": tool_artifacts,
        }
        (artifact_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        relative_dir = Path(session_id) / "artifacts" / parent_run_id / agent_id
        audit = {
            "type": "subagent_audit",
            "thread_id": payload["thread_id"],
            "agent_id": agent_id,
            "run_id": payload["run_id"],
            "status": payload["status"],
            "artifact_id": artifact_id,
            "summary": payload["summary"],
            "manifest_path": (relative_dir / "manifest.json").as_posix(),
            "handoff_path": (relative_dir / "handoff.json").as_posix(),
            "final_answer_path": (relative_dir / "final_answer.md").as_posix(),
            "tool_result_count": len(payload["tool_results"]),
            "tool_names": handoff["tool_names"],
            "step_count": payload["step_count"],
        }
        return ToolExecutionResult(
            call=tool_result.call,
            output=json.dumps(audit, ensure_ascii=False),
            truncated_output=tool_result.truncated_output,
            error=tool_result.error,
            annotations=tool_result.annotations,
        )
