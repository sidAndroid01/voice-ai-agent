"""A scripted LLM so agent behaviour can be tested offline and deterministically."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Union

from livekit.agents import llm
from livekit.agents.llm import ChatChunk, ChoiceDelta, FunctionToolCall
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions


class ToolCall:
    def __init__(self, name: str, **arguments: Any) -> None:
        self.name = name
        self.arguments = arguments


Step = Union[str, ToolCall, Callable[[llm.ChatContext], Union[str, ToolCall]]]


class ScriptedLLM(llm.LLM):
    """Each chat() call consumes the next step: reply text or a tool call."""

    def __init__(self, steps: list[Step]) -> None:
        super().__init__()
        self.steps = list(steps)
        self.requests: list[llm.ChatContext] = []

    @property
    def model(self) -> str:
        return "scripted"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **_: Any,
    ) -> llm.LLMStream:
        self.requests.append(chat_ctx.copy())
        step: Step = self.steps.pop(0) if self.steps else "Okay."
        if callable(step) and not isinstance(step, ToolCall):
            step = step(chat_ctx)
        return _Stream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options, step=step)


class _Stream(llm.LLMStream):
    def __init__(self, llm_: ScriptedLLM, *, step: str | ToolCall, **kw: Any) -> None:
        super().__init__(llm_, **kw)
        self._step = step

    async def _run(self) -> None:
        if isinstance(self._step, ToolCall):
            delta = ChoiceDelta(
                role="assistant",
                tool_calls=[
                    FunctionToolCall(
                        name=self._step.name,
                        arguments=json.dumps(self._step.arguments),
                        call_id=f"call_{id(self._step)}",
                    )
                ],
            )
        else:
            delta = ChoiceDelta(role="assistant", content=self._step)
        self._event_ch.send_nowait(ChatChunk(id="chunk", delta=delta))
