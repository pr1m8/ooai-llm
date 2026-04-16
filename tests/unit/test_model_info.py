"""Unit tests for ergonomic model-info helpers."""

from __future__ import annotations

import sys
import types

from ooai_llm import AppSettings, get_model_info


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


class FakeLLM:
    model = "openai:gpt-5.4-mini"
    profile = {
        "max_input_tokens": 100,
        "tool_calling": True,
        "tool_choice": True,
        "structured_output": True,
        "reasoning_output": True,
    }
    disabled_params = {}

    def get_num_tokens_from_messages(self, messages, tools=None):
        return len(messages) * 10


class FakeNoParallelLLM(FakeLLM):
    disabled_params = {"parallel_tool_calls": None}



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



def test_get_model_info_from_llm_infers_parallel_tool_calls():
    settings = AppSettings()
    info = get_model_info(llm=FakeLLM(), settings=settings)

    assert info.identity.litellm_model == "openai/gpt-5.4-mini"
    assert info.capabilities.parallel_tool_calls is True
    assert info.capabilities.tool_choice is True



def test_get_model_info_respects_disabled_parallel_tool_calls():
    info = get_model_info(llm=FakeNoParallelLLM(), settings=AppSettings())
    assert info.capabilities.parallel_tool_calls is False



def test_get_model_info_with_messages_adds_message_estimate():
    _install_fake_langchain_messages()
    info = get_model_info(
        llm=FakeLLM(),
        settings=AppSettings(),
        messages=[{"role": "user", "content": "hello"}, {"role": "user", "content": "again"}],
    )

    assert info.message_estimate is not None
    assert info.message_estimate.estimated_input_tokens == 20
    assert info.message_estimate.fits_context_window is True
