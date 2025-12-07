import asyncio
from contextlib import suppress
from typing import AsyncIterator
from uuid import uuid4

from .errors import AceAIRuntimeError
from .event_bus import IEventBus, InMemoryEventBus
from .executor import ToolExecutor
from .llm import LLMMessage, LLMResponse, LLMService
from .llm.models import LLMToolCallMessage, LLMToolUseMessage
from .models import AgentResponse, AgentStep, AgentStepEvent, ToolExecutionResult


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
        event_bus: IEventBus | None = None,
    ):
        self.prompt = prompt
        self.default_model = default_model
        self.llm_service = llm_service
        self.executor = executor
        self.max_turns = max_turns
        self._event_bus: IEventBus = event_bus or InMemoryEventBus()
        self.agent_registry[self.__class__.__name__] = self

    async def handle(
        self,
        question: str,
        *,
        model: str | None = None,
    ) -> AgentResponse:
        """Run the agent to completion and return the structured trace."""

        try:
            return await self._run_agent(
                question,
                model=model,
                event_bus=self._event_bus,
            )
        finally:
            await self._event_bus.close()

    async def stream(
        self, question: str, *, model: str | None = None
    ) -> AsyncIterator[AgentStepEvent]:
        """Stream AgentStepEvent entries as the agent reasons."""

        subscription = self._event_bus.subscribe()

        async def runner() -> None:
            try:
                await self._run_agent(
                    question,
                    model=model,
                    event_bus=self._event_bus,
                )
            finally:
                await self._event_bus.close()

        task = asyncio.create_task(runner())

        try:
            async for event in subscription:
                yield event
                if event.event_type == "agent.run.completed":
                    break
            await task
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            close = getattr(subscription, "aclose", None)
            if close is not None:
                with suppress(Exception):
                    await close()

    async def _run_agent(
        self,
        question: str,
        *,
        model: str | None,
        event_bus: IEventBus,
    ) -> AgentResponse:
        messages: list[LLMMessage] = [
            LLMMessage(role="system", content=self.prompt),
            LLMMessage(role="user", content=question),
        ]
        selected_model = model or self.default_model
        turns: list[AgentStep] = []

        async def publish(event: AgentStepEvent) -> None:
            await event_bus.publish(event)

        for _ in range(self.max_turns):
            step_index = len(turns)
            step_id = str(uuid4())
            await publish(
                AgentStepEvent(
                    event_type="agent.llm.started",
                    step_index=step_index,
                    step_id=step_id,
                )
            )
            response: LLMResponse = await self.llm_service.complete(
                messages=messages,
                tools=self.executor.tool_schemas,
                metadata={"model": selected_model},
            )
            step = AgentStep(step_id=step_id, llm_response=response)
            turns.append(step)
            await publish(
                AgentStepEvent(
                    event_type="agent.llm.completed",
                    step_index=step_index,
                    step_id=step_id,
                    step=step,
                )
            )

            if response.tool_calls:
                assistant_msg = LLMToolCallMessage(
                    content=response.text,
                    tool_calls=response.tool_calls,
                )
                messages.append(assistant_msg)
                final_answer_output: str | None = None

                for call in response.tool_calls:
                    await publish(
                        AgentStepEvent(
                            event_type="agent.tool.started",
                            step_index=step_index,
                            step_id=step_id,
                            tool_call=call,
                        )
                    )
                    try:
                        tool_output = await self.executor.execute_tool(call)
                    except Exception as exc:
                        error_msg = str(exc)
                        failed_result = ToolExecutionResult(call=call, error=error_msg)
                        step.tool_results.append(failed_result)
                        await publish(
                            AgentStepEvent(
                                event_type="agent.tool.failed",
                                step_index=step_index,
                                step_id=step_id,
                                tool_call=call,
                                tool_result=failed_result,
                                error=error_msg,
                            )
                        )
                        await publish(
                            AgentStepEvent(
                                event_type="agent.step.failed",
                                step_index=step_index,
                                step_id=step_id,
                                step=step,
                                error=error_msg,
                            )
                        )
                        await publish(
                            AgentStepEvent(
                                event_type="agent.run.completed",
                                step_index=step_index,
                                step_id=step_id,
                                step=step,
                                error=error_msg,
                            )
                        )
                        raise

                    tool_result = ToolExecutionResult(call=call, output=tool_output)
                    step.tool_results.append(tool_result)
                    await publish(
                        AgentStepEvent(
                            event_type="agent.tool.completed",
                            step_index=step_index,
                            step_id=step_id,
                            tool_call=call,
                            tool_result=tool_result,
                        )
                    )

                    if call.name == "final_answer":
                        final_answer_output = tool_result.output
                        break

                    messages.append(
                        LLMToolUseMessage(
                            name=call.name,
                            call_id=call.call_id,
                            content=tool_result.output,
                        )
                    )

                await publish(
                    AgentStepEvent(
                        event_type="agent.step.completed",
                        step_index=step_index,
                        step_id=step_id,
                        step=step,
                    )
                )

                if final_answer_output is not None:
                    await publish(
                        AgentStepEvent(
                            event_type="agent.run.completed",
                            step_index=step_index,
                            step_id=step_id,
                            step=step,
                            annotations={"final_output": final_answer_output},
                        )
                    )
                    return AgentResponse(turns=turns, final_output=final_answer_output)

                continue

            final_answer = response.text.strip()

            await publish(
                AgentStepEvent(
                    event_type="agent.step.completed",
                    step_index=step_index,
                    step_id=step_id,
                    step=step,
                )
            )

            if final_answer:
                await publish(
                    AgentStepEvent(
                        event_type="agent.run.completed",
                        step_index=step_index,
                        step_id=step_id,
                        step=step,
                        annotations={"final_output": final_answer},
                    )
                )
                return AgentResponse(turns=turns, final_output=final_answer)

            messages.append(LLMMessage(role="assistant", content=""))

        error_msg = "Agent exceeded maximum reasoning turns without answering"
        if turns:
            last_step = turns[-1]
            last_index = len(turns) - 1
            await publish(
                AgentStepEvent(
                    event_type="agent.step.failed",
                    step_index=last_index,
                    step_id=last_step.step_id,
                    step=last_step,
                    error=error_msg,
                )
            )
            await publish(
                AgentStepEvent(
                    event_type="agent.run.completed",
                    step_index=last_index,
                    step_id=last_step.step_id,
                    step=last_step,
                    error=error_msg,
                )
            )
        else:
            await publish(
                AgentStepEvent(
                    event_type="agent.run.completed",
                    step_index=0,
                    step_id=str(uuid4()),
                    error=error_msg,
                )
            )
        raise AceAIRuntimeError(error_msg)
