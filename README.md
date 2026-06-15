# Advisor/Executor Strategy

A minimal, runnable proof-of-concept for a **model-agnostic**, **tool-agnostic** coding agent with separate advisor and executor roles.

📖 **Read the architecture guide**: [abster12.github.io/advisor-executor-strategy](https://abster12.github.io/advisor-executor-strategy)

## What it demonstrates

- **Advisor/Executor split**: advisor plans, executor calls tools.
- **Model-agnostic routing**: swap providers per role (OpenAI, Anthropic, Ollama/mock).
- **Tool-agnostic layer**: built-in tools + MCP server discovery.
- **CLI + config-driven**: runs without API keys in mock mode.

## Install

```bash
cd advisor-executor-strategy
uv pip install -e ".[all]"
# or: pip install -e ".[all]"
```

For mock-only mode, base dependencies are enough:

```bash
uv pip install -e .
```

## Run in mock mode

```bash
python -m advisor_executor_poc "Read a file and run a shell command" --config config.yaml
```

The mock advisor emits a canned plan; the mock executor calls the real built-in tools.

## Run with a local model (Ollama)

If Ollama is running on `localhost:11434`, use `config-ollama.yaml`:

```bash
python -m advisor_executor_poc "List files in the current directory" --config config-ollama.yaml
```

This uses the OpenAI-compatible endpoint at `http://localhost:11434/v1` and requires no API key.

## Run with cloud providers

1. Set API keys:

```bash
export OPENAI_API_KEY=***
export ANTHROPIC_API_KEY=***
```

2. Edit `config.yaml`:

```yaml
models:
  advisor:
    provider: anthropic
    model: claude-sonnet-4-20250514
    reasoning_effort: high
  executor:
    provider: openai
    model: gpt-4o
```

3. Run:

```bash
python -m advisor_executor_poc "Summarize the project structure" --config config.yaml
```

## Use MCP servers

Uncomment an `mcp_servers` entry in `config.yaml` or use `config-mcp.yaml` (requires Node.js for `npx`):

```bash
python -m advisor_executor_poc "What time is it?" --config config-mcp.yaml
```

This discovers tools from the MCP server and registers them as `mcp_{server}_{tool}`. The MCP connection runs in a background thread so synchronous tool calls from the agent kernel work without deadlock.

```yaml
mcp_servers:
  time:
    command: uvx
    args: ["mcp-server-time"]
```

## Use as a plugin (MCP server)

The PoC can expose itself as an MCP server so any host can call it without a custom plugin:

```bash
python -m advisor_executor_poc --stdio --config config.yaml
```

This exposes one tool: `aepoc_run_task(request: str, context_json?: str)`.

### VS Code

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "advisor-executor-poc": {
      "command": "python",
      "args": ["-m", "advisor_executor_poc", "--stdio", "--config", "config.yaml"]
    }
  }
}
```

### Cursor / Claude Desktop / Claude Code

Add to the host's MCP server config:

```json
{
  "mcpServers": {
    "advisor-executor-poc": {
      "command": "python",
      "args": ["-m", "advisor_executor_poc", "--stdio", "--config", "config.yaml"]
    }
  }
}
```

### Hermes Agent

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  aepoc:
    command: "python"
    args: ["-m", "advisor_executor_poc", "--stdio", "--config", "config.yaml"]
```

## Project layout

```
advisor_executor_poc/
├── config.py        # Config loading
├── models.py        # Model router + provider adapters
├── tools.py         # Tool registry + built-in/MCP tools
├── mcp_client.py    # Sync wrapper around async MCP client
├── mcp_server.py    # MCP server exposing the agent kernel
├── plan.py          # Plan/Step data structures
├── agent.py         # Advisor + Executor + AgentKernel
└── cli.py           # CLI entry point
```

## Next steps to harden

1. Add streaming responses and usage tracking.
2. Add structured output constraints for the advisor (`response_format: json_schema`).
3. Add persistent state store (SQLite) for long-running plans.
4. Add approval UI per tool call.
5. Add more provider adapters (Gemini, Bedrock, Azure OpenAI).
