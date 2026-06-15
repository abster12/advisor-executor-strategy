"""Tool registry with built-in tools and optional MCP integration."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from advisor_executor_poc.config import McpServerConfig


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: dict


@dataclass
class ToolResult:
    success: bool
    output: str


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable[[dict], ToolResult]] = {}
        self._schemas: dict[str, ToolSchema] = {}
        self._register_builtin_tools()
        self._mcp_clients: dict[str, Any] = {}

    def _register_builtin_tools(self) -> None:
        self.register(
            name="read_file",
            description="Read the contents of a file.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path"}
                },
                "required": ["path"],
            },
            handler=self._read_file,
        )
        self.register(
            name="list_directory",
            description="List files in a directory.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"}
                },
                "required": ["path"],
            },
            handler=self._list_directory,
        )
        self.register(
            name="run_command",
            description="Run a shell command and return stdout/stderr.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"}
                },
                "required": ["command"],
            },
            handler=self._run_command,
        )

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        handler: Callable[[dict], ToolResult],
    ) -> None:
        self._schemas[name] = ToolSchema(name, description, parameters)
        self._tools[name] = handler

    @property
    def schemas(self) -> list[ToolSchema]:
        return list(self._schemas.values())

    def call(self, name: str, arguments: dict) -> ToolResult:
        if name not in self._tools:
            return ToolResult(False, f"Tool '{name}' not found.")
        try:
            return self._tools[name](arguments)
        except Exception as e:
            return ToolResult(False, f"Tool error: {e}")

    # ------------------------------------------------------------------
    # Built-in handlers
    # ------------------------------------------------------------------

    def _read_file(self, args: dict) -> ToolResult:
        path = Path(args["path"]).expanduser()
        if not path.exists():
            return ToolResult(False, f"File not found: {path}")
        try:
            return ToolResult(True, path.read_text(encoding="utf-8", errors="replace"))
        except Exception as e:
            return ToolResult(False, str(e))

    def _list_directory(self, args: dict) -> ToolResult:
        path = Path(args.get("path", ".")).expanduser()
        if not path.exists():
            return ToolResult(False, f"Directory not found: {path}")
        try:
            entries = "\n".join(str(p) for p in sorted(path.iterdir()))
            return ToolResult(True, entries)
        except Exception as e:
            return ToolResult(False, str(e))

    def _run_command(self, args: dict) -> ToolResult:
        command = args["command"]
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout
            if result.stderr:
                output += "\n[stderr]\n" + result.stderr
            return ToolResult(result.returncode == 0, output)
        except subprocess.TimeoutExpired:
            return ToolResult(False, "Command timed out after 30s")
        except Exception as e:
            return ToolResult(False, str(e))

    # ------------------------------------------------------------------
    # MCP integration (optional)
    # ------------------------------------------------------------------

    async def connect_mcp_servers(self, configs: dict[str, McpServerConfig]) -> None:
        """Discover and register tools from MCP servers.

        Requires `pip install -e '.[mcp]'`.
        """
        try:
            from advisor_executor_poc.mcp_client import SyncMcpClient
        except ImportError:
            return

        for name, cfg in configs.items():
            if not cfg.command:
                continue
            try:
                client = SyncMcpClient(name, cfg)
                client.start()
                tools = client.list_tools()
                for tool in tools.tools:
                    prefixed = f"mcp_{name}_{tool.name}"
                    self.register(
                        name=prefixed,
                        description=tool.description or "",
                        parameters=tool.inputSchema,
                        handler=self._make_mcp_handler(client, tool.name),
                    )
                self._mcp_clients[name] = client
            except Exception as e:
                print(f"[WARN] Failed to connect MCP server '{name}': {e}")

    def close(self) -> None:
        """Close MCP sessions."""
        for client in self._mcp_clients.values():
            try:
                client.stop()
            except Exception:
                pass
        self._mcp_clients.clear()

    def _make_mcp_handler(
        self, client: Any, tool_name: str
    ) -> Callable[[dict], ToolResult]:
        def handler(args: dict) -> ToolResult:
            try:
                result = client.call_tool(tool_name, args)
                text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
                return ToolResult(not result.isError, text)
            except Exception as e:
                return ToolResult(False, f"MCP call error: {repr(e)}")

        return handler
