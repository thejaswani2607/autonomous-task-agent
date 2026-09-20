import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from state import AgentState

load_dotenv()
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

MODEL_NAME = "gemini-3.1-flash-lite"

# Tool declarations — this is what tells Gemini what actions exist and what arguments each needs.
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
        description="Run Python code in an isolated sandbox to compute, organize, merge, or transform data gathered so far. Use print() to output results.",
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
        description="Save content to a file on disk as the final step, once the data is ready. Use .csv for CSV files, .xlsx for Excel files.",
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
Do not repeat an action that already failed in the same way - try a different approach instead.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[TOOLS],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            ),
        ),
    )

    part = response.candidates[0].content.parts[0]
    if part.function_call:
        return {
            "tool_name": part.function_call.name,
            "tool_args": dict(part.function_call.args),
        }

    # Shouldn't normally happen since mode="ANY" forces a function call every time
    return {"tool_name": None, "tool_args": {}, "raw_text": response.text or ""}