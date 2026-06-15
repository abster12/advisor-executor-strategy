"""Model router and provider adapters.

The kernel talks to models through a single OpenAI-style interface.
Adapters translate to/from provider-native SDKs.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from advisor_executor_poc.config import ModelConfig


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: dict


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ChatResponse:
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    model: str | None = None
    usage: dict | None = None


class ModelAdapter(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        config: ModelConfig | None = None,
    ) -> ChatResponse:
        raise NotImplementedError


class MockAdapter(ModelAdapter):
    """Deterministic adapter for testing the loop without API keys."""

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        config: ModelConfig | None = None,
    ) -> ChatResponse:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )

        # Advisor mode: emit a plan when asked to plan.
        if config and "advisor" in config.model:
            steps = [
                {
                    "action": "read_file",
                    "target": "example.txt",
                    "description": "Read the target file",
                    "purpose": "Understand current content",
                },
                {
                    "action": "run_command",
                    "command": "echo 'hello from executor'",
                    "description": "Run a verification command",
                    "purpose": "Verify executor works",
                },
            ]
            if "time" in last_user.lower():
                steps = [
                    {
                        "action": "mcp_time_get_current_time",
                        "description": "Get current UTC time",
                        "purpose": "Answer the user's time question",
                        "arguments": {"timezone": "UTC"},
                    }
                ]
            plan_json = json.dumps(
                {
                    "goal": last_user,
                    "context": {"notes": "Mock context"},
                    "steps": steps,
                },
                indent=2,
            )
            return ChatResponse(
                content=f"```plan\n{plan_json}\n```",
                model="mock-advisor",
            )

        # Executor mode: emit a tool call or a summary.
        if tools:
            first = tools[0]
            args = {k: "mock" for k in first.parameters.get("properties", {})}
            # Parse step from prompt to use real target/command when possible.
            step_match = re.search(
                r"Execute this step:\s*\n(\{.*?\})\n\nCall exactly one tool",
                last_user or "",
                re.DOTALL,
            )
            if step_match:
                try:
                    step = json.loads(step_match.group(1))
                    action = step.get("action", "")
                    target = step.get("target") or step.get("path")
                    command = step.get("command")
                    explicit_args = step.get("arguments") or {}
                    if explicit_args:
                        args = explicit_args
                    else:
                        if "path" in args and target:
                            args["path"] = target
                        if "command" in args and command:
                            args["command"] = command
                    # Prefer the tool whose name matches the action if possible.
                    matching = [t for t in tools if t.name == action]
                    if matching:
                        first = matching[0]
                        if explicit_args:
                            args = explicit_args
                        else:
                            args = {k: target if k == "path" else command if k == "command" else "mock" for k in first.parameters.get("properties", {})}
                            if "path" in args and not target:
                                args["path"] = "mock"
                            if "command" in args and not command:
                                args["command"] = "echo mock"
                except json.JSONDecodeError:
                    pass
            return ChatResponse(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name=first.name,
                        arguments=args,
                    )
                ],
                model="mock-executor",
            )

        return ChatResponse(content="Mock executor summary.", model="mock-executor")


class OpenAIAdapter(ModelAdapter):
    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        config: ModelConfig | None = None,
    ) -> ChatResponse:
        from openai import OpenAI

        cfg = config or ModelConfig()
        api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY")
        # Ollama/vLLM/local OpenAI-compatible endpoints don't require an API key.
        if api_key is None:
            api_key = "sk-dummy"
        client = OpenAI(
            api_key=api_key,
            base_url=cfg.base_url,
        )

        openai_messages = []
        for m in messages:
            item: dict = {"role": m.role, "content": m.content}
            if m.role == "assistant" and m.tool_calls:
                item["tool_calls"] = m.tool_calls
            if m.role == "tool":
                item["tool_call_id"] = m.tool_call_id
                item["name"] = m.name
            openai_messages.append(item)

        openai_tools = None
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        kwargs: dict = {
            "model": cfg.model,
            "messages": openai_messages,
        }
        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature
        if openai_tools:
            kwargs["tools"] = openai_tools

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in choice.message.tool_calls
            ]
        return ChatResponse(
            content=choice.message.content,
            tool_calls=tool_calls,
            model=response.model,
            usage=response.usage.to_dict() if response.usage else None,
        )


class AnthropicAdapter(ModelAdapter):
    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        config: ModelConfig | None = None,
    ) -> ChatResponse:
        from anthropic import Anthropic

        cfg = config or ModelConfig()
        client = Anthropic(
            api_key=cfg.api_key or os.environ.get("ANTHROPIC_API_KEY"),
            base_url=cfg.base_url,
        )

        system = "\n".join(m.content for m in messages if m.role == "system")
        conversation = []
        for m in messages:
            if m.role == "user":
                conversation.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                conversation.append({"role": "assistant", "content": m.content})
            elif m.role == "tool":
                conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": m.content,
                            }
                        ],
                    }
                )

        anthropic_tools = None
        if tools:
            anthropic_tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]

        kwargs: dict = {
            "model": cfg.model,
            "max_tokens": 4096,
            "messages": conversation,
        }
        if system:
            kwargs["system"] = system
        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        if cfg.reasoning_effort:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 16000}

        response = client.messages.create(**kwargs)
        tool_calls = None
        content_blocks = [b for b in response.content if b.type == "text"]
        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        content = "\n".join(b.text for b in content_blocks) or None
        if tool_blocks:
            tool_calls = [
                ToolCall(id=b.id, name=b.name, arguments=b.input)
                for b in tool_blocks
            ]
        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            if response.usage
            else None,
        )


class ModelRouter:
    """Routes role-based calls to the configured provider adapter."""

    def __init__(self, configs: dict[str, ModelConfig]):
        self.configs = configs
        self._adapters: dict[str, ModelAdapter] = {}

    def _adapter_for(self, provider: str) -> ModelAdapter:
        provider = provider.lower()
        if provider == "mock":
            return MockAdapter()
        if provider == "openai":
            return OpenAIAdapter()
        if provider in ("anthropic", "claude"):
            return AnthropicAdapter()
        raise ValueError(f"Unknown provider: {provider}")

    def chat(
        self,
        role: str,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> ChatResponse:
        cfg = self.configs.get(role)
        if cfg is None:
            raise ValueError(f"No model configured for role: {role}")
        adapter = self._adapter_for(cfg.provider)
        return adapter.chat(messages, tools=tools, config=cfg)
