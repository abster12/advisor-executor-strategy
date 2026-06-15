"""CLI entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from advisor_executor_poc.config import Config
from advisor_executor_poc.agent import AgentKernel


DEFAULT_CONFIG_PATHS = [
    Path.home() / ".config" / "aepoc" / "config.yaml",
    Path.cwd() / "config.yaml",
]


def _find_config() -> Path | None:
    for path in DEFAULT_CONFIG_PATHS:
        if path.exists():
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model-agnostic, tool-agnostic advisor/executor agent PoC"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="{}",
        help="JSON context object",
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Run as an MCP server over stdio instead of the CLI",
    )
    parser.add_argument(
        "request",
        nargs="?",
        default=None,
        help="User request (or read from stdin)",
    )
    args = parser.parse_args()

    if args.stdio:
        from advisor_executor_poc.mcp_server import run_server
        asyncio.run(run_server(args.config))
        return

    if args.config:
        config_path = Path(args.config).expanduser()
    else:
        found = _find_config()
        if found:
            config_path = found
        else:
            print("No config found; using defaults (mock provider).", file=sys.stderr)
            config_path = None

    config = Config.from_file(config_path) if config_path else Config()

    if args.request:
        request = args.request
    else:
        print("Enter request (Ctrl-D to finish):")
        request = sys.stdin.read().strip()

    try:
        context = json.loads(args.context)
    except json.JSONDecodeError:
        print("Invalid --context JSON", file=sys.stderr)
        sys.exit(1)

    kernel = AgentKernel(config)
    asyncio.run(kernel.connect_tools())
    try:
        plan = kernel.run(request, context)
    finally:
        kernel.tools.close()

    print("\n" + "=" * 60)
    print(f"Plan status: {plan.status}")
    if plan.final_result:
        print(f"Result: {plan.final_result}")
    for step in plan.steps:
        icon = "✓" if step.status == "done" else "✗" if step.status == "failed" else "o"
        print(f"{icon} Step {step.id}: {step.description}")


if __name__ == "__main__":
    main()
