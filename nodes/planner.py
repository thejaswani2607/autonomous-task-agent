import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from state import AgentState
from llm_utils import call_with_retry

load_dotenv()
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="web_search",
        description="Search the web for information. Use this to find facts, prices, comparisons, or any information not already known.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "The search query"}
            },
            "required": ["query"],
        },
    ),
    types.FunctionDeclaration(
        name="run_python",
        description="Run Python code in an isolated sandbox. Use this ONLY when you need actual computation - math, sorting, merging numeric data, deduplication logic. Do NOT use this just to format text into CSV rows - write the CSV text yourself and call write_file directly instead.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "code": {"type": "STRING", "description": "Python code to execute"}
            },
            "required": ["code"],
        },
    ),
    types.FunctionDeclaration(
        name="write_file",
        description="Save content to a file on disk as the final step, once the data is ready. Use .csv for CSV files, .xlsx for Excel files. You can write the CSV-formatted content directly yourself - no need to use run_python first just to build simple text.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "path": {"type": "STRING", "description": "File path, e.g. output.csv or output.xlsx"},
                "content": {"type": "STRING", "description": "CSV-formatted text content to write"},
            },
            "required": ["path", "content"],
        },
    ),
])


def _format_history(state: AgentState) -> str:
    """Turn the history list into readable text for the prompt."""
    if not state.history:
        return "No actions taken yet."
    lines = []
    for s in state.history:
        line = f"Step {s.step_number}: called {s.tool_name}({s.tool_args}) -> "
        line += "SUCCESS" if s.success else "FAILED"
        line += f": {s.result[:300]}"
        if s.evaluator_verdict:
            line += f" | Evaluator said: {s.evaluator_verdict} - {s.evaluator_reasoning}"
        lines.append(line)
    return "\n".join(lines)


def plan_next_action(state: AgentState) -> dict:
    """
    Ask Gemini to decide the ONE next tool call, given the goal and history so far.
    Returns: {"tool_name": ..., "tool_args": {...}}
    """
    prompt = f"""You are the planning module of an autonomous agent.

GOAL: {state.goal}

HISTORY SO FAR:
{_format_history(state)}

Decide the SINGLE next tool call that makes the most progress toward the goal.

Efficiency rules - follow these strictly:
- Do NOT repeat an action that already failed in the same way - try a different approach instead.
- Be decisive: if the goal needs N items (e.g. "3 tools", "3 recipes") and you can already
  identify N distinct, usable items from the searches done so far, STOP searching immediately
  and move to saving the file. Do not keep searching "to be thorough" once you have enough.
- Prefer calling write_file DIRECTLY with the final CSV-formatted text you compose yourself.
  Only use run_python first if you genuinely need to compute something (math, sorting, merging
  numeric values) - not just to format plain text into rows.
- Every extra step costs real time and money, so the fewest steps that correctly satisfy the
  goal is always the best plan.
"""

    response = call_with_retry(lambda model: client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[TOOLS],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            ),
        ),
    ))

    part = response.candidates[0].content.parts[0]
    if part.function_call:
        return {
            "tool_name": part.function_call.name,
            "tool_args": dict(part.function_call.args),
        }

    return {"tool_name": None, "tool_args": {}, "raw_text": response.text or ""}