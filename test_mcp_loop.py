"""Manual test: run the advisor/executor loop with an MCP tool step."""

import asyncio

from advisor_executor_poc.config import Config
from advisor_executor_poc.agent import AgentKernel
from advisor_executor_poc.plan import Plan, Step


def main():
    cfg = Config.from_file("config-mcp.yaml")
    kernel = AgentKernel(cfg)
    asyncio.run(kernel.connect_tools())

    # Inject a plan that uses an MCP tool.
    plan = Plan(
        goal="Get current UTC time",
        steps=[
            Step(
                action="mcp_time_get_current_time",
                description="Get current UTC time",
                purpose="Answer the user",
                arguments={"timezone": "UTC"},
            )
        ],
    )

    while not plan.is_complete():
        step = plan.next_step()
        result = kernel.executor.execute_step(step)
        print(f"[{step.action}] success={result.success}")
        print(result.output)
        plan.advance(result.output, success=result.success)

    kernel.tools.close()


if __name__ == "__main__":
    main()
