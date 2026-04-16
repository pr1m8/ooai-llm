"""Message normalization and lightweight message estimation helpers.

Purpose:
    Normalize common message inputs into both LangChain-style message objects
    and OpenAI/LiteLLM-style message dictionaries.

Design:
    - Accept ergonomic inputs such as a bare string, a sequence of mapping
      objects, or an existing sequence of LangChain messages.
    - Convert inputs lazily so importing :mod:`ooai_llm` does not require the
      full LangChain runtime until message handling is used.
    - Keep the public surface small and provider-agnostic so the same helpers
      can later be reused for embeddings, tool traces, and other higher-level
      features.

Examples:
    >>> normalized = normalize_messages("hello")
    >>> normalized.openai_messages[0]["role"]
    'user'
    >>> normalized.message_count
    1
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

MessageMapping: TypeAlias = Mapping[str, Any]
MessageLike: TypeAlias = str | MessageMapping | Any
MessagesLike: TypeAlias = str | Sequence[MessageLike]

_ROLE_TO_CLASS_NAME: dict[str, str] = {
    "user": "HumanMessage",
    "human": "HumanMessage",
    "system": "SystemMessage",
    "assistant": "AIMessage",
    "ai": "AIMessage",
    "tool": "ToolMessage",
}

_MESSAGE_TYPE_TO_ROLE: dict[str, str] = {
    "human": "user",
    "user": "user",
    "system": "system",
    "ai": "assistant",
    "assistant": "assistant",
    "tool": "tool",
}


class NormalizedMessages(BaseModel):
    """Normalized message payload in LangChain and OpenAI-compatible forms.

    Args:
        langchain_messages: LangChain-style message objects.
        openai_messages: OpenAI/LiteLLM-style message dictionaries.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    langchain_messages: list[Any] = Field(default_factory=list)
    openai_messages: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def message_count(self) -> int:
        """Return the number of normalized messages."""
        return len(self.openai_messages)


class MessageEstimate(BaseModel):
    """Optional message-derived estimate attached to model info.

    Args:
        message_count: Number of normalized messages.
        estimated_input_tokens: Estimated prompt/input tokens when available.
        fits_context_window: Whether the estimated prompt fits the model's
            context window when both values are known.
        warning: Optional best-effort warning or limitation note.
    """

    model_config = ConfigDict(frozen=True)

    message_count: int = 0
    estimated_input_tokens: int | None = None
    fits_context_window: bool | None = None
    warning: str | None = None


def _message_imports() -> dict[str, Any]:
    """Import LangChain message classes lazily.

    Returns:
        Mapping of commonly used message classes.

    Raises:
        ImportError: If ``langchain_core`` is not installed.
    """
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

    return {
        "AIMessage": AIMessage,
        "BaseMessage": BaseMessage,
        "HumanMessage": HumanMessage,
        "SystemMessage": SystemMessage,
        "ToolMessage": ToolMessage,
    }


def _coerce_mapping_to_langchain(message: MessageMapping) -> Any:
    """Convert a mapping with a ``role`` key to a LangChain message object."""
    imports = _message_imports()
    role = str(message.get("role", "user")).strip().lower()
    class_name = _ROLE_TO_CLASS_NAME.get(role)
    if class_name is None:
        raise ValueError(f"Unsupported message role: {role!r}.")
    content = message.get("content", "")
    if class_name == "ToolMessage":
        tool_call_id = str(
            message.get("tool_call_id")
            or message.get("tool_call")
            or message.get("id")
            or "tool_call"
        )
        return imports[class_name](content=content, tool_call_id=tool_call_id)
    if class_name == "AIMessage":
        kwargs: dict[str, Any] = {}
        if "tool_calls" in message:
            kwargs["tool_calls"] = message["tool_calls"]
        if "name" in message:
            kwargs["name"] = message["name"]
        return imports[class_name](content=content, **kwargs)
    return imports[class_name](content=content)


def _coerce_to_langchain_message(message: MessageLike) -> Any:
    """Normalize a single message-like value into a LangChain message."""
    if isinstance(message, str):
        return _message_imports()["HumanMessage"](content=message)
    if isinstance(message, Mapping):
        return _coerce_mapping_to_langchain(message)

    imports = _message_imports()
    if isinstance(message, imports["BaseMessage"]):
        return message

    raise TypeError(f"Unsupported message value: {message!r}.")


def _langchain_to_openai_message(message: Any) -> dict[str, Any]:
    """Convert a LangChain message object into an OpenAI-compatible dict."""
    message_type = getattr(message, "type", None) or message.__class__.__name__.replace("Message", "").lower()
    role = _MESSAGE_TYPE_TO_ROLE.get(str(message_type).lower(), "user")
    payload: dict[str, Any] = {
        "role": role,
        "content": getattr(message, "content", ""),
    }

    if role == "assistant":
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            payload["tool_calls"] = tool_calls
    if role == "tool":
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            payload["tool_call_id"] = tool_call_id
    name = getattr(message, "name", None)
    if name:
        payload["name"] = name
    return payload


def normalize_messages(messages: MessagesLike) -> NormalizedMessages:
    """Normalize message input into LangChain and OpenAI-compatible forms.

    Args:
        messages: Bare prompt string, message mapping, or sequence of either.

    Returns:
        Normalized message payload.
    """
    sequence: Sequence[MessageLike]
    if isinstance(messages, str):
        sequence = [messages]
    elif isinstance(messages, Mapping):
        sequence = [messages]
    else:
        sequence = list(messages)

    langchain_messages = [_coerce_to_langchain_message(message) for message in sequence]
    openai_messages = [_langchain_to_openai_message(message) for message in langchain_messages]
    return NormalizedMessages(
        langchain_messages=langchain_messages,
        openai_messages=openai_messages,
    )
