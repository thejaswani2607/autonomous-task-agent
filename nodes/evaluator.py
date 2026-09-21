import os
import json
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


def evaluate_progress(state: AgentState) -> AgentState:
    """
    Ask Gemini to judge overall progress toward the goal, given the full history.
    Updates the last history entry with the verdict, and sets state.done /
    state.final_answer if the goal is fully complete. Also enforces the
    15-step hard safety cap regardless of what Gemini says.
    """
    prompt = f"""You are the evaluator module of an autonomous agent.

GOAL: {state.goal}

FULL HISTORY:
{_format_history(state)}

Judge the current progress:
- "done": the goal has been FULLY achieved (all required outputs actually exist, e.g. a file was actually written successfully)
- "continue": progress is being made but the goal isn't fully done yet
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