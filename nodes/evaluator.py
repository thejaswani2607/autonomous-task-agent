import os
import json
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv
from state import AgentState
from llm_utils import call_with_retry

load_dotenv()
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "verdict": {"type": "STRING", "enum": ["continue", "done", "stuck"]},
        "reasoning": {"type": "STRING"},
        "final_answer": {"type": "STRING"},
    },
    "required": ["verdict", "reasoning"],
}

FILE_OUTPUT_HINTS = ["save", "csv", "excel", "xlsx", ".csv", ".xlsx", "file", "spreadsheet"]


def _format_history(state: AgentState) -> str:
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


def _extract_required_item_count(goal: str) -> int | None:
    """
    Look for a small number in the goal that likely specifies how many
    distinct items are needed (e.g. "3 recipes", "find 5 tools").
    Returns None if no plausible count is found. Bounded to 2-20 to avoid
    misfiring on unrelated numbers (like a year or a filename).
    """
    matches = re.findall(r"\b(\d+)\b", goal)
    for m in matches:
        n = int(m)
        if 2 <= n <= 20:
            return n
    return None


def _successful_search_count(state: AgentState) -> int:
    return sum(1 for s in state.history if s.tool_name == "web_search" and s.success)


def evaluate_progress(state: AgentState) -> AgentState:
    """
    Ask Gemini to judge overall progress toward the goal, given the full history.
    Updates the last history entry with the verdict, and sets state.done /
    state.final_answer if the goal is fully complete. Enforces the 15-step
    hard safety cap, plus two code-level checks that override "done" when
    the LLM's own judgment can't be trusted alone: (1) a file must actually
    have been saved if the goal implies one, and (2) at least as many
    successful searches as any item-count mentioned in the goal.
    """
    prompt = f"""You are the evaluator module of an autonomous agent.

GOAL: {state.goal}

FULL HISTORY:
{_format_history(state)}

Break the goal down into EVERY separate concrete requirement, including:
- Any specific QUANTITY mentioned (e.g. "3 recipes", "5 tools", "top 10") - count how many
  genuinely distinct, named items actually appear across the history. If the goal asked for
  N items and fewer than N distinct items are clearly present, this requirement is NOT satisfied,
  even if a file was saved.
- Any file-saving requirement (e.g. "save as", "csv", "excel") - separate from the above.

Check the history against EACH requirement individually before deciding. Be strict about counts -
do not assume a requirement is met just because a plausible-sounding file exists.

Judge the current progress:
- "done": ALL requirements are fully satisfied, with actual successful tool calls proving each one
- "continue": progress is being made but at least one requirement isn't satisfied yet
- "stuck": the last action(s) failed and a different approach is needed

If verdict is "done", write a short final_answer summarizing what was accomplished.
"""

    response = call_with_retry(lambda model: client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        ),
    ))

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

    # --- Code-level safety net #1: file must actually be saved if implied ---
    if verdict == "done" and _goal_requires_file_output(state.goal) and not _has_successful_write_file(state):
        verdict = "continue"
        reasoning = (
            "Not yet done: the goal requires a file saved via write_file specifically, "
            "and that hasn't succeeded yet, regardless of what other progress was made."
        )

    # --- Code-level safety net #2: enough searches to plausibly cover the requested count ---
    required_count = _extract_required_item_count(state.goal)
    if verdict == "done" and required_count is not None:
        search_count = _successful_search_count(state)
        if search_count < required_count:
            verdict = "continue"
            reasoning = (
                f"Not yet done: the goal appears to require {required_count} distinct items, "
                f"but only {search_count} successful search(es) have been done so far - "
                f"not enough evidence that {required_count} distinct items were actually found."
            )

    if state.history:
        state.history[-1].evaluator_verdict = verdict
        state.history[-1].evaluator_reasoning = reasoning

    if verdict == "done":
        state.done = True
        state.final_answer = final_answer or "Goal completed."

    if state.step_count >= 15 and not state.done:
        state.done = True
        state.final_answer = "Stopped: reached the 15-step safety limit before the goal was confirmed complete."

    return state