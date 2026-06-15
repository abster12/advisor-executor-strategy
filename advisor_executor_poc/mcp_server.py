"""Expose the agent kernel as an MCP server over stdio.

This lets any MCP host (VS Code, Cursor, Claude Code, Zed, Hermes, etc.)
invoke the advisor/executor agent as a tool without writing a custom plugin.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from advisor_executor_poc.cli import DEFAULT_CONFIG_PATHS
from advisor_executor_poc.config import Config
from advisor_executor_poc.agent import AgentKernel


async def run_server(config_path: str | None = None) -> None:
    server = Server("advisor-executor-poc")

    if config_path:
        cfg_path = config_path
    else:
        cfg_path = next((str(p) for p in DEFAULT_CONFIG_PATHS if p.exists()), None)

    config = Config.from_file(cfg_path) if cfg_path else Config()
    kernel = AgentKernel(config)
    await kernel.connect_tools()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="aepoc_run_task",
                description=(
                    "Run a task through the advisor/executor agent. "
                    "The advisor plans steps and the executor runs tools. "
                    "Returns the final status and step results."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "request": {
                            "type": "string",
                            "description": "The user request to fulfill",
                        },
                        "context_json": {
                            "type": "string",
                            "description": "Optional JSON object with context (relevant files, constraints, etc.)",
                        },
                    },
                    "required": ["request"],
                },
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name != "aepoc_run_task":
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        request = arguments.get("request", "")
        context = {}
        context_json = arguments.get("context_json")
        if context_json:
            try:
                context = json.loads(context_json)
            except json.JSONDecodeError:
                return [TextContent(type="text", text="Invalid context_json")]

        # Run the synchronous kernel in a thread so it doesn't block the MCP loop.
        loop = asyncio.get_event_loop()
        plan = await loop.run_in_executor(None, lambda: kernel.run(request, context))

        lines = [f"Plan status: {plan.status}", ""]
        for step in plan.steps:
            status_icon = "✓" if step.status == "done" else "✗" if step.status == "failed" else "o"
            lines.append(f"{status_icon} Step {step.id}: {step.description}")
            if step.observation:
                obs = step.observation[:500]
                if len(step.observation) > 500:
                    obs += " ..."
                lines.append(f"   {obs}")
        return [TextContent(type="text", text="\n".join(lines))]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

    kernel.tools.close()
