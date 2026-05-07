import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from aceai.agent.ace_agent import build_ace_agent
from aceai.core.events import RunCompletedEvent, ToolCompletedEvent


E2E_ENV_FILE = Path(__file__).parent / "e2e.env"
DELEGATION_SENTINEL = "ACEAI_DELEGATION_E2E_42F9"

load_dotenv(E2E_ENV_FILE)


@pytest.mark.e2e
@pytest.mark.live_llm
@pytest.mark.anyio
@pytest.mark.skipif(
    "DEEPSEEK_API_KEY" not in os.environ,
    reason="DEEPSEEK_API_KEY is required for live DeepSeek e2e tests",
)
async def test_ace_agent_delegates_task_with_live_deepseek_v4_pro() -> None:
    agent = build_ace_agent(
        provider_name="deepseek",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        model="deepseek-v4-pro",
        skill_path="disable",
        tool_enabled={
            "list_directory": False,
            "read_text_file": False,
            "write_text_file": False,
            "replace_text_in_file": False,
            "preview_patch": False,
            "apply_patch": False,
            "git_status": False,
            "git_diff": False,
            "run_shell_command": False,
            "search_text": False,
        },
    )

    final_answer = ""
    delegation_outputs: list[dict[str, object]] = []
    async for event in agent.run(
        "This is an AceAI live delegation e2e test. You must call the "
        "`delegate_to_subagent` tool exactly once before answering. Delegate this task: "
        f"return the sentinel `{DELEGATION_SENTINEL}` and the calculation "
        "`19 + 23 = 42`. Use these delegate_to_subagent arguments: "
        "task='Return the sentinel and calculation result.', "
        "instructions='Return a concise Summary/Evidence/Risks result. Include "
        f"the exact sentinel {DELEGATION_SENTINEL} and the equation 19 + 23 = 42.', "
        "context_brief='This test verifies that a child agent can run independently "
        "and return a result to the main agent.', allowed_tools=[]. After the tool "
        "returns, answer with the same sentinel and equation."
    ):
        if isinstance(event, ToolCompletedEvent) and event.tool_name == "delegate_to_subagent":
            delegation_outputs.append(json.loads(event.tool_result.output))
        elif isinstance(event, RunCompletedEvent):
            final_answer = event.final_answer

    assert len(delegation_outputs) == 1
    child_result = delegation_outputs[0]
    assert child_result["status"] == "completed"
    assert DELEGATION_SENTINEL in child_result["final_answer"]
    assert "42" in child_result["final_answer"]
    assert DELEGATION_SENTINEL in final_answer
    assert "42" in final_answer
