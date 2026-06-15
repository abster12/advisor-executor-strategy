"""Plan/observation data structures used by advisor and executor."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


class Step(BaseModel):
    id: int = 0
    action: str
    target: str | None = None
    command: str | None = None
    arguments: dict[str, Any] | None = None
    description: str
    purpose: str | None = None
    status: StepStatus = StepStatus.PENDING
    observation: str | None = None


class Plan(BaseModel):
    goal: str
    context: dict[str, Any] = Field(default_factory=dict)
    steps: list[Step] = Field(default_factory=list)
    current_step_index: int = 0
    status: str = "in_progress"  # in_progress | complete | failed
    final_result: str | None = None

    @model_validator(mode="after")
    def _assign_step_ids(self) -> "Plan":
        for i, step in enumerate(self.steps, start=1):
            if step.id == 0:
                step.id = i
        return self

    def is_complete(self) -> bool:
        return self.status in ("complete", "failed")

    def next_step(self) -> Step | None:
        if self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def advance(self, observation: str, success: bool = True) -> None:
        step = self.steps[self.current_step_index]
        step.observation = observation
        step.status = StepStatus.DONE if success else StepStatus.FAILED
        self.current_step_index += 1
        if self.current_step_index >= len(self.steps):
            self.status = "complete" if success else "failed"

    def revise_steps(self, steps: list[Step]) -> None:
        """Replace remaining/pending steps, keeping completed ones."""
        completed = self.steps[: self.current_step_index]
        for i, step in enumerate(steps, start=self.current_step_index + 1):
            step.id = i
        self.steps = completed + steps
        self._assign_step_ids()
