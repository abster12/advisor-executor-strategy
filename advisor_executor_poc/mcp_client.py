"""Synchronous wrapper around an async MCP stdio client.

Runs the MCP connection in a dedicated background thread with its own event
loop so that tools can be called synchronously from the agent kernel.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Any, Callable

from advisor_executor_poc.config import McpServerConfig


class SyncMcpClient:
    """Connects to one MCP server and exposes sync `call_tool`/`list_tools`."""

    def __init__(self, name: str, cfg: McpServerConfig):
        self.name = name
        self.cfg = cfg
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Any = None
        self._stdio_ctx: Any = None
        self._session_ctx: Any = None
        self._ready = threading.Event()
        self._closed = False

    def start(self) -> None:
        def _run_loop() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._connect())
            self._ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=_run_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=30):
            raise TimeoutError(f"MCP server '{self.name}' failed to start")

    async def _connect(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.cfg.command,
            args=self.cfg.args,
            env={**dict(__import__("os").environ), **self.cfg.env} if self.cfg.env else None,
        )
        self._stdio_ctx = stdio_client(params)
        read, write = await self._stdio_ctx.__aenter__()
        self._session_ctx = ClientSession(read, write)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()

    def list_tools(self) -> list[Any]:
        return self._run_async(lambda: self._session.list_tools())

    def call_tool(self, tool_name: str, arguments: dict) -> Any:
        return self._run_async(lambda: self._session.call_tool(tool_name, arguments=arguments))

    def _run_async(self, coro: Callable[[], Any]) -> Any:
        if self._closed or self._loop is None:
            raise RuntimeError("MCP client is not running")
        future: Future[Any] = asyncio.run_coroutine_threadsafe(coro(), self._loop)
        return future.result(timeout=60)

    def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._run_async(self._shutdown)
        except Exception:
            pass
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)

    async def _shutdown(self) -> None:
        try:
            await self._session_ctx.__aexit__(None, None, None)
        except Exception:
            pass
        try:
            await self._stdio_ctx.__aexit__(None, None, None)
        except Exception:
            pass
