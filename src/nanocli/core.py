"""Core agent loop and data model for nanocli.

This module defines the minimal data model (messages, tool outputs, tool requests)
and the agent loop that drives the Responses API + bash-only tool execution.
"""

import hashlib
import re
import subprocess
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from openai import OpenAI
from openai.types.responses import EasyInputMessageParam, ResponseInputItemParam, ResponseReasoningItemParam

# Types used throughout nanocli:
# - Message/Reasoning/BashOutput are the only items persisted in the conversation context.
# - BashRequest is the only "tool call" signal emitted by the loop for approval.
# This keeps the model-facing API and the CLI loop intentionally simple.


@dataclass(frozen=True)
class Message:
    """A plain chat message stored in the conversation context."""

    role: Literal["system", "developer", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class Reasoning:
    """Bridges the Responses API reasoning item into the chat context."""

    id: str
    # these fields are to conveniently bridge the Responses API's Reasoning type
    summary: list[object]
    content: list[object]
    encrypted_content: str | None


@dataclass(frozen=True)
class BashOutput:
    """Structured output captured from running a bash command."""

    stdout: str
    stderr: str
    returncode: int

    def format(self) -> str:
        return f"""
<stdout>
{self.stdout}
</stdout>

<stderr>
{self.stderr}
</stderr>

<returncode>
{self.returncode}
</returncode>
""".strip()


type ChatItem = Message | Reasoning | BashOutput
type ChatContext = list[ChatItem]


@dataclass(frozen=True)
class BashRequest:
    """Signal to request user approval before executing a bash command."""

    command: str


type TurnEvent = ChatItem | BashRequest


@dataclass
class Session:
    """Holds session configuration and conversation context for a single chat."""

    model: str
    instructions: str
    instructions_as_system: bool = False
    yolo: bool = False
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "none"
    cwd: Path = field(default_factory=Path.cwd)
    context: ChatContext = field(default_factory=list)
    client: OpenAI = field(default_factory=OpenAI)

    def __post_init__(self) -> None:
        """Derive a stable prompt cache key from the instructions."""
        digest = hashlib.sha256(self.instructions.encode("utf-8")).hexdigest()
        self.prompt_cache_key = f"nanocli:{digest}"

    def stream_response_final(self) -> list[ChatItem]:
        """Call the Responses API and return the final list of output items."""

        def to_response_api(chat_item: ChatItem) -> ResponseInputItemParam:
            match chat_item:
                case Message(role, content):
                    return EasyInputMessageParam(role=role, content=content)
                case Reasoning(id, summary, content, encrypted_content):
                    return ResponseReasoningItemParam(
                        id=id,
                        summary=summary,  # type: ignore
                        content=content,  # type: ignore
                        encrypted_content=encrypted_content,
                        type="reasoning",
                    )
                case BashOutput() as output:
                    return EasyInputMessageParam(role="user", content=output.format())

        input_items: list[ResponseInputItemParam] = []
        if self.instructions_as_system:
            input_items.append(EasyInputMessageParam(role="system", content=self.instructions))
        input_items.extend(map(to_response_api, self.context))
        extra_args: dict = {}
        if self.reasoning_effort != "none":
            extra_args["reasoning"] = dict(effort=self.reasoning_effort, summary="auto")
        if not self.instructions_as_system:
            extra_args["instructions"] = self.instructions
        stream = self.client.responses.create(
            model=self.model,
            input=input_items,
            include=["reasoning.encrypted_content"],
            stream=True,
            store=False,
            prompt_cache_key=self.prompt_cache_key,
            **extra_args,
        )
        for event in stream:
            if event.type == "response.completed":
                outputs: list[ChatItem] = []
                for output in event.response.output:
                    if output.type == "message":
                        for content in output.content:
                            if hasattr(content, "text"):
                                outputs.append(Message(output.role, content.text))  # type: ignore[arg-type]
                    elif output.type == "reasoning":
                        outputs.append(Reasoning(output.id, output.summary, output.content, output.encrypted_content))  # type: ignore
                return outputs
        raise AssertionError("Stream ended without response.completed.")


def parse_bash_block(text: str) -> str | None:
    """Return the first fenced ```bash block, or None if absent."""
    # STUDENT TODO: extract only fenced bash blocks.
    raise NotImplementedError


def run_bash_command(command: str, cwd: Path) -> BashOutput:
    """Run a shell command and return structured output."""
    # STUDENT TODO: implement shell execution.
    raise NotImplementedError


def run_turn(session: Session, user_text: str) -> Generator[TurnEvent, bool | None, None]:
    """Execute one user turn with a bash-only tool loop."""
    # STUDENT TODO: implement the agentic loop and bash-only tool parsing.
    raise NotImplementedError
