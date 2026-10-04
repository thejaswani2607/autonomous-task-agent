import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from state import AgentState
from llm_utils import call_with_retry
from checks import goal_requires_file_output, extract_required_item_count

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


def _has_successful_write_file(state: AgentState) -> bool:
    return any(s.tool_name == "write_file" and s.success for s in state.history)


def _successful_search_count(state: AgentState) -> int:
    return sum(1 for s in state.history if s.tool_name == "web_search" and s.success)


def evaluate_progress(state: AgentState) -> AgentState:
    """
    Ask Gemini to judge overall progress toward the goal, given the full history.
    Updates the last history entry with the verdict, and sets state.done /
    state.final_answer if the goal is fully complete. Enforces the 15-step
    hard safety cap, plus two code-level checks that override "done" when
    the LLM's own judgment can't be trusted alone: (1) a file must actually
    have been saved if the goal asks for one, and (2) at least as many
    successful searches as any item-count mentioned in the goal.
    """
    prompt = f"""You are the evaluator module of an autonomous agent.

GOAL: {state.goal}

FULL HISTORY:
{_format_history(state)}

Break the goal down into EVERY separate concrete requirement, including:
- Any specific QUANTITY mentioned (e.g. "3 recipes", "5 tools", "top 10") - count how many
  genuinely distinct, named items actually appear across the history, backed by real web_search
  results (not just the agent's own stated claims). If the goal asked for N items and fewer than
  N distinct items are clearly present with search evidence, this requirement is NOT satisfied.
- A file-saving requirement - ONLY if the goal explicitly asks for a saved file (e.g. "save as",
  "csv", "excel", "file"). If the goal does not ask for a file, do NOT require one.
  IMPORTANT: write_file automatically adds a timestamp to the filename if a file with that name
  already exists, to avoid overwriting previous output. A successful write_file call satisfies
  the file-saving requirement even if the saved filename doesn't exactly match what was requested -
  do not reject completion just because of an auto-added timestamp in the filename.
  A write_file step marked BLOCKED means the goal did not ask for a file - ignore it.

Check the history against EACH requirement individually before deciding. Be strict about counts -
do not assume a requirement is met just because a plausible-sounding file exists.

CRITICAL for final_answer: it must be SPECIFIC and must answer what the goal actually asked.
Only name specific items (tools, recipes, products, hotels, etc.) that appear in successful
web_search results in the history - never use your own general knowledge or an earlier assumption.
If the history does not yet contain enough evidence to answer specifically, do NOT return "done"
with a vague answer - return "continue" so that more searching happens.

If the goal asks for ONE best option, a recommendation, or "which one" (for example "tell me the
best option", "recommend", "which should I buy"), then "done" requires that final_answer names
exactly ONE specific pick, with a short reason taken from the search results. A list of several
models is NOT an answer to that kind of goal.

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

    # --- Code-level safety net #1: a file must actually be saved if the goal asks for one ---
    if verdict == "done" and goal_requires_file_output(state.goal) and not _has_successful_write_file(state):
        verdict = "continue"
        reasoning = (
            "Not yet done: the goal requires a file saved via write_file specifically, "
            "and that hasn't succeeded yet, regardless of what other progress was made."
        )

    # --- Code-level safety net #2: enough searches to plausibly cover the requested count ---
    required_count = extract_required_item_count(state.goal)
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