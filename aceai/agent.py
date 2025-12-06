from .executor import ToolExecutor
from .llm import LLMMessage, LLMResponse, LLMService
from .llm.models import LLMToolCallMessage, LLMToolUseMessage


class AgentBase:
    """Base class for agents using an LLM provider.

    Required dependencies are injected explicitly (no optional defaults).
    """

    agent_registry: dict[str, "AgentBase"] = {}

    def __init__(
        self,
        *,
        prompt: str,
        default_model: str,
        llm_service: LLMService,
        executor: ToolExecutor,
        max_turns: int = 5,
    ):
        self.prompt = prompt
        self.default_model = default_model
        self.llm_service = llm_service
        self.executor = executor
        self.max_turns = max_turns
        self.agent_registry[self.__class__.__name__] = self

    async def handle(self, question: str, *, model: str | None = None) -> str:
        "TODO: this should be async generator"

        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=self.prompt),
            LLMMessage(role="user", content=question),
        ]
        selected_model = model or self.default_model

        for _ in range(self.max_turns):
            response: LLMResponse = await self.llm_service.complete(
                messages=messages,
                tools=self.executor.tool_schemas,
                metadata={"model": selected_model},
            )

            if response.tool_calls:
                assistant_msg = LLMToolCallMessage(
                    content=response.text,
                    tool_calls=response.tool_calls,
                )
                messages.append(assistant_msg)
                for call in response.tool_calls:
                    tool_result = await self.executor.execute_tool(call)
                    if call.name == "final_answer":
                        return tool_result

                    messages.append(
                        LLMToolUseMessage(
                            name=call.name,
                            call_id=call.call_id,
                            content=tool_result,
                        )
                    )
                continue

            final_answer = response.text.strip()
            if final_answer:
                return final_answer
            messages.append(LLMMessage(role="assistant", content=""))

        raise RuntimeError("Agent exceeded maximum reasoning turns without answering")
