"""Configuration loading and validation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


def _resolve_env(value: Any) -> Any:
    """Recursively resolve `os.environ/VAR_NAME` strings."""
    if isinstance(value, str) and value.startswith("os.environ/"):
        return os.environ.get(value[len("os.environ/"):], "")
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


class ModelConfig(BaseModel):
    provider: str = "mock"
    model: str = "mock"
    temperature: float | None = 0.1
    reasoning_effort: str | None = None
    api_key: str | None = None
    base_url: str | None = None


class McpServerConfig(BaseModel):
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)


class ApprovalConfig(BaseModel):
    mode: str = "manual"  # manual | smart | off
    auto_approve: list[str] = Field(default_factory=list)
    prompt_on: list[str] = Field(default_factory=list)


class Config(BaseModel):
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    mcp_servers: dict[str, McpServerConfig] = Field(default_factory=dict)
    approval: ApprovalConfig = Field(default_factory=ApprovalConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        path = Path(path).expanduser()
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        raw = _resolve_env(raw)
        # YAML `mcp_servers:` with no value becomes None; normalize to empty dicts.
        for key in ("mcp_servers", "models", "approval"):
            if raw.get(key) is None:
                raw[key] = {}
        return cls(**raw)
