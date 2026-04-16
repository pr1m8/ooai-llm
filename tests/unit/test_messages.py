"""Unit tests for message normalization helpers."""

from __future__ import annotations

import sys
import types

from ooai_llm.messages import normalize_messages


class _BaseMessage:
    def __init__(self, content, **kwargs):
        self.content = content
        for key, value in kwargs.items():
            setattr(self, key, value)


class HumanMessage(_BaseMessage):
    type = "human"


class SystemMessage(_BaseMessage):
    type = "system"


class AIMessage(_BaseMessage):
    type = "ai"


class ToolMessage(_BaseMessage):
    type = "tool"



def _install_fake_langchain_messages() -> None:
    langchain_core = types.ModuleType("langchain_core")
    messages = types.ModuleType("langchain_core.messages")
    messages.BaseMessage = _BaseMessage
    messages.HumanMessage = HumanMessage
    messages.SystemMessage = SystemMessage
    messages.AIMessage = AIMessage
    messages.ToolMessage = ToolMessage
    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.messages"] = messages



def test_normalize_messages_from_string(monkeypatch):
    _install_fake_langchain_messages()

    normalized = normalize_messages("hello")

    assert normalized.message_count == 1
    assert normalized.openai_messages == [{"role": "user", "content": "hello"}]
    assert isinstance(normalized.langchain_messages[0], HumanMessage)



def test_normalize_messages_from_role_mappings(monkeypatch):
    _install_fake_langchain_messages()

    normalized = normalize_messages(
        [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "tool_calls": [{"name": "foo", "args": {}, "id": "1"}]},
            {"role": "tool", "content": "done", "tool_call_id": "1"},
        ]
    )

    assert normalized.message_count == 4
    assert normalized.openai_messages[0]["role"] == "system"
    assert normalized.openai_messages[2]["tool_calls"][0]["name"] == "foo"
    assert normalized.openai_messages[3]["tool_call_id"] == "1"
