from pydantic import BaseModel
from typing import Optional, Literal


class StepRecord(BaseModel):
    """One single round of the loop: what the Planner chose, what happened."""
    step_number: int
    tool_name: str                # which tool was called, e.g. "web_search"
    tool_args: dict                # the arguments given to that tool
    result: str                    # what came back (or the error message)
    success: bool                  # did it run without error?
    evaluator_verdict: Optional[Literal["continue", "done", "stuck"]] = None
    evaluator_reasoning: Optional[str] = None


class AgentState(BaseModel):
    """The full shared memory that flows through every node in the graph."""
    goal: str
    history: list[StepRecord] = []
    step_count: int = 0
    done: bool = False
    final_answer: Optional[str] = None

    # for the cost/step tracker improvement
    llm_call_count: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0

    # temporary holding spot: Planner writes here, Executor reads and clears it
    pending_action: Optional[dict] = None

    def pretty(self) -> str:
        """Human-readable multi-line view of the state, for terminal output."""
        lines = [
            f"Goal: {self.goal}",
            f"Step: {self.step_count} | Done: {self.done}",
            f"LLM calls: {self.llm_call_count} | Tool successes: {self.successful_tool_calls} | Tool failures: {self.failed_tool_calls}",
            "History:",
        ]
        if not self.history:
            lines.append("  (no steps yet)")
        else:
            for step in self.history:
                status = "✅ success" if step.success else "❌ failed"
                lines.append(f"  [{step.step_number}] {step.tool_name}({step.tool_args}) → {status}")
                lines.append(f"      result: {step.result}")
                if step.evaluator_verdict:
                    lines.append(f"      evaluator: {step.evaluator_verdict} — {step.evaluator_reasoning}")
        if self.final_answer:
            lines.append(f"Final answer: {self.final_answer}")
        return "\n".join(lines)