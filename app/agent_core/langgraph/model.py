"""Chat model integration for the LangGraph agent foundation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from openai import OpenAI

from app.agent_core.groq_provider import GroqLLMProviderAdapter
from app.ai.provider import get_groq_client
from app.core.config import Settings

# ``groq_provider`` adds the standalone Agent Orchestrator directory to the
# import path, so these imports are intentionally kept after it.
from agent_orchestrator.exceptions import ProviderError
from agent_orchestrator.providers.models import ProviderRequest


class ChatModelProviderError(RuntimeError):
    """Safe provider failure raised by the chat model adapter."""


class ChatModelRunner(Protocol):
    """Minimal model invocation contract used by the graph node."""

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Any] | None = None,
    ) -> AIMessage:
        """Invoke the model and return an assistant message.

        ``tools`` contains the application-owned tool definitions that should be
        advertised for this request. Implementations may ignore it when no tools
        are available.
        """


def _to_provider_message(message: BaseMessage | dict[str, Any]) -> dict[str, Any]:
    """Convert LangGraph/LangChain messages to the existing Groq adapter format."""
    if isinstance(message, dict):
        return message

    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}

    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}

    if isinstance(message, AIMessage):
        provider_message: dict[str, Any] = {
            "role": "assistant",
            "content": message.content,
        }
        if message.tool_calls:
            provider_message["tool_calls"] = [
                {
                    "id": tool_call.get("id"),
                    "call_id": tool_call.get("id"),
                    "name": tool_call.get("name"),
                    "arguments": tool_call.get("args"),
                }
                for tool_call in message.tool_calls
            ]
        return provider_message

    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "name": message.name,
            "content": message.content,
        }

    return {
        "role": getattr(message, "role", "user"),
        "content": getattr(message, "content", ""),
    }


class GroqChatModelRunner:
    """Run the configured Groq model through the existing provider adapter."""

    def __init__(self, client: OpenAI, model_name: str):
        self._adapter = GroqLLMProviderAdapter(client, model_name)

    def invoke(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[Any] | None = None,
    ) -> AIMessage:
        provider_messages = [_to_provider_message(message) for message in messages]
        try:
            response = self._adapter.complete(
                ProviderRequest(
                    messages=provider_messages,
                    tools=list(tools) if tools else None,
                ),
            )
        except ProviderError:
            raise ChatModelProviderError("Groq provider request failed.") from None

        tool_calls = [
            {
                "name": tool_call.name,
                "args": tool_call.arguments,
                "id": tool_call.id,
            }
            for tool_call in response.tool_calls
        ]
        return AIMessage(content=response.content or "", tool_calls=tool_calls)


def build_groq_chat_model(settings: Settings) -> GroqChatModelRunner:
    """Build a Groq chat model runner from the application settings."""
    client = get_groq_client(settings)
    if client is None:
        raise ValueError("Groq is not configured: GROQ_API_KEY is required.")
    if not settings.groq_model:
        raise ValueError("GROQ_MODEL must be set when GROQ_API_KEY is provided.")
    return GroqChatModelRunner(client, settings.groq_model)