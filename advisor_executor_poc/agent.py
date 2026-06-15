"""Advisor/Executor kernel."""

from __future__ import annotations

import json
import re
from typing import Any

from advisor_executor_poc.config import Config
from advisor_executor_poc.models import Message, ModelRouter, ToolCall, ToolSchema
from advisor_executor_poc.plan import Plan, Step, StepStatus
from advisor_executor_poc.tools import ToolRegistry, ToolResult


ADVISOR_SYSTEM_PROMPT = """You are an expert software engineering advisor.
Your job is to plan how to fulfill the user's request.
You do NOT execute tools yourself; you produce a structured plan for an executor.

Output your plan inside a JSON code block labeled `plan`. The plan must follow this schema:

{
  "goal": "concise restatement of the user request",
  "context": {
    "relevant_files": ["optional/file1.py", "optional/file2.py"],
    "constraints": ["any constraints or notes"]
  },
  "steps": [
    {
      "action": "read_file | list_directory | run_command | <mcp_server_tool_name>",
      "target": "file or directory path (for file/dir tools)",
      "command": "shell command (for run_command)",
      "description": "what this step does",
      "purpose": "why this step is needed"
    }
  ]
}

Use only tools that exist in the executor's tool registry. If you need to revise the plan based on executor feedback, output a new `plan` block with the remaining steps."""


EXECUTOR_SYSTEM_PROMPT = """You are a precise executor.
You receive a plan step and must execute it by calling exactly one tool.
Do not explain; call the tool with valid JSON arguments.
If the step is already complete or no tool is needed, respond with a concise summary."""


class Advisor:
    def __init__(self, router: ModelRouter):
        self.router = router

    def plan(self, request: str, context: dict[str, Any]) -> Plan:
        messages = [
            Message(role="system", content=ADVISOR_SYSTEM_PROMPT),
            Message(
                role="user",
                content=f"User request: {request}\n\nContext: {json.dumps(context, indent=2)}",
            ),
        ]
        response = self.router.chat("advisor", messages)
        return self._parse_plan(response.content or "", request, context)

    def revise(self, plan: Plan, observation: str) -> Plan:
        """Ask advisor to revise remaining steps based on an observation."""
        remaining = plan.steps[plan.current_step_index :]
        messages = [
            Message(role="system", content=ADVISOR_SYSTEM_PROMPT),
            Message(
                role="user",
                content=(
                    f"Original goal: {plan.goal}\n"
                    f"Completed steps: {plan.current_step_index}\n"
                    f"Remaining steps: {json.dumps([s.model_dump() for s in remaining], indent=2)}\n"
                    f"Latest observation: {observation}\n\n"
                    "Revise the remaining steps if needed. Output a new `plan` JSON block."
                ),
            ),
        ]
        response = self.router.chat("advisor", messages)
        new_plan = self._parse_plan(response.content or "", plan.goal, plan.context)
        plan.revise_steps(new_plan.steps)
        return plan

    def _parse_plan(self, content: str, goal: str, context: dict) -> Plan:
        # Extract JSON from ```plan ... ``` block.
        match = re.search(r"```plan\s*\n(.*?)\n```", content, re.DOTALL)
        if match:
            json_text = match.group(1)
        else:
            # Fallback: look for any JSON object.
            match = re.search(r"\{.*\}", content, re.DOTALL)
            json_text = match.group(0) if match else "{}"
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            data = {}
        steps = [Step(**s) for s in data.get("steps", [])]
        return Plan(
            goal=data.get("goal", goal),
            context=data.get("context", context),
            steps=steps,
        )


class Executor:
    def __init__(self, router: ModelRouter, tools: ToolRegistry):
        self.router = router
        self.tools = tools

    def execute_step(self, step: Step) -> ToolResult:
        step.status = StepStatus.IN_PROGRESS
        tool_schemas = [
            ToolSchema(name=s.name, description=s.description, parameters=s.parameters)
            for s in self.tools.schemas
        ]

        messages = [
            Message(role="system", content=EXECUTOR_SYSTEM_PROMPT),
            Message(
                role="user",
                content=self._step_to_prompt(step, tool_schemas),
            ),
        ]
        response = self.router.chat("executor", messages, tools=tool_schemas)

        if response.tool_calls:
            result = self._execute_tool_call(response.tool_calls[0])
            return result

        # No tool call: assume step is informational.
        return ToolResult(True, response.content or "No action taken.")

    def _step_to_prompt(self, step: Step, schemas: list[ToolSchema]) -> str:
        tools_text = json.dumps(
            [
                {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.parameters,
                }
                for s in schemas
            ],
            indent=2,
        )
        step_json = step.model_dump_json(indent=2)
        return (
            f"Available tools:\n{tools_text}\n\n"
            f"Execute this step:\n{step_json}\n\n"
            "Call exactly one tool with valid arguments. If the step includes an `arguments` field, use those values."
        )

    def _execute_tool_call(self, call: ToolCall) -> ToolResult:
        return self.tools.call(call.name, call.arguments)


class AgentKernel:
    def __init__(self, config: Config):
        self.config = config
        self.router = ModelRouter(config.models)
        self.tools = ToolRegistry()
        self.advisor = Advisor(self.router)
        self.executor = Executor(self.router, self.tools)

    async def connect_tools(self) -> None:
        if self.config.mcp_servers:
            await self.tools.connect_mcp_servers(self.config.mcp_servers)

    def run(self, request: str, context: dict[str, Any] | None = None) -> Plan:
        plan = self.advisor.plan(request, context or {})
        max_steps = 20
        last_failed_step_signature = None
        while not plan.is_complete() and max_steps > 0:
            step = plan.next_step()
            if step is None:
                plan.status = "complete"
                break
            print(f"\n[Step {step.id}] {step.description}")
            result = self.executor.execute_step(step)
            print(f"[Result] success={result.success}")
            print(result.output[:500])
            if len(result.output) > 500:
                print("...")
            plan.advance(result.output, success=result.success)
            if not result.success and self.config.approval.mode != "off":
                revised = self.advisor.revise(plan, result.output)
                # Avoid infinite loops if advisor keeps emitting the same failed step.
                signature = _step_signature(revised.steps[revised.current_step_index :])
                if signature == last_failed_step_signature:
                    plan.status = "failed"
                    break
                last_failed_step_signature = signature
                plan = revised
            max_steps -= 1
        return plan


def _step_signature(steps: list[Step]) -> tuple:
    return tuple((s.action, s.target, s.command, s.description) for s in steps)
