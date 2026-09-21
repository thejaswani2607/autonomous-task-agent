import os
import json
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv
from state import AgentState

load_dotenv()
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

MODEL_NAME = "gemini-3.1-flash-lite"

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "verdict": {"type": "STRING", "enum": ["continue", "done", "stuck"]},
        "reasoning": {"type": "STRING"},
        "final_answer": {"type": "STRING"},
    },
    "required": ["verdict", "reasoning"],
}

# Keywords that signal the goal expects a real file to be saved.
FILE_OUTPUT_HINTS = ["save", "csv", "excel", "xlsx", ".csv", ".xlsx", "file", "spreadsheet"]


def _format_history(state: AgentState) -> str:
    """Turn the history list into readable text for the prompt."""
    if not state.history:
        return "No actions taken yet."
    lines = []
    for s in state.history:
        line = f"Step {s.step_number}: called {s.tool_name}({s.tool_args}) -> "
        line += "SUCCESS" if s.success else "FAILED"
        line += f": {s.result[:400]}"
        lines.append(line)
    return "\n".join(lines)


def _goal_requires_file_output(goal: str) -> bool:
    goal_lower = goal.lower()
    return any(hint in goal_lower for hint in FILE_OUTPUT_HINTS)


def _has_successful_write_file(state: AgentState) -> bool:
    return any(s.tool_name == "write_file" and s.success for s in state.history)


def evaluate_progress(state: AgentState) -> AgentState:
    """
    Ask Gemini to judge overall progress toward the goal, given the full history.
    Updates the last history entry with the verdict, and sets state.done /
    state.final_answer if the goal is fully complete. Enforces the 15-step
    hard safety cap, and a code-level check that refuses "done" if the goal
    clearly needed a saved file but write_file never succeeded.
    """
    prompt = f"""You are the evaluator module of an autonomous agent.

GOAL: {state.goal}

FULL HISTORY:
{_format_history(state)}

Break the goal down into EVERY separate concrete requirement (e.g. "find N items" is
one requirement, "save as a file" is a SEPARATE requirement). Check the history against
EACH requirement individually before deciding.

Judge the current progress:
- "done": ALL requirements are fully satisfied, with actual successful tool calls proving each one
- "continue": progress is being made but at least one requirement isn't satisfied yet
- "stuck": the last action(s) failed and a different approach is needed

If verdict is "done", write a short final_answer summarizing what was accomplished.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        ),
    )

    try:
        verdict_data = json.loads(response.text)
    except (json.JSONDecodeError, TypeError):
        verdict_data = {
            "verdict": "continue",
            "reasoning": "Evaluator response could not be parsed; defaulting to continue.",
        }

    verdict = verdict_data.get("verdict", "continue")
    reasoning = verdict_data.get("reasoning", "")
    final_answer = verdict_data.get("final_answer")

    # --- Code-level safety net: don't trust "done" blindly ---
    if verdict == "done" and _goal_requires_file_output(state.goal) and not _has_successful_write_file(state):
        verdict = "continue"
        reasoning = (
            "Overridden by safety check: the goal requires a saved file, but no successful "
            "write_file call exists in history yet. " + reasoning
        )

    if state.history:
        state.history[-1].evaluator_verdict = verdict
        state.history[-1].evaluator_reasoning = reasoning

    if verdict == "done":
        state.done = True
        state.final_answer = final_answer or "Goal completed."

    # Hard safety cap — stop regardless of verdict once we hit 15 steps
    if state.step_count >= 15 and not state.done:
        state.done = True
        state.final_answer = "Stopped: reached the 15-step safety limit before the goal was confirmed complete."

    return state