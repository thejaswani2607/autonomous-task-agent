import subprocess
import tempfile
import os
import io
from datetime import datetime
from pydantic import BaseModel
from tavily import TavilyClient
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# ---------- Argument schemas (used for LLM function-calling definitions) ----------

class WebSearchArgs(BaseModel):
    query: str


class RunPythonArgs(BaseModel):
    code: str


class WriteFileArgs(BaseModel):
    path: str
    content: str


# ---------- Tool implementations ----------

def web_search(query: str) -> str:
    """Search the web and return a readable summary of top results."""
    response = tavily_client.search(query=query, max_results=5)
    results = response.get("results", [])
    if not results:
        return "No results found."

    lines = []
    for r in results:
        lines.append(f"- {r.get('title')}: {r.get('content', '')[:300]} (source: {r.get('url')})")
    return "\n".join(lines)


def run_python(code: str, timeout: int = 10) -> str:
    """
    Run Python code in an isolated subprocess with a timeout.
    This is the sandboxing layer: the code never runs inside our main
    program's memory — it's a separate, disposable process we can kill safely.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return f"ERROR (exit code {result.returncode}): {result.stderr.strip()}"
        return result.stdout.strip() if result.stdout.strip() else "(code ran successfully, no output printed)"
    except subprocess.TimeoutExpired:
        return f"ERROR: code timed out after {timeout} seconds"
    finally:
        os.remove(tmp_path)


def _make_unique_path(path: str) -> str:
    """
    If path already exists, insert a timestamp before the extension
    so we never silently overwrite a previous run's output.
    e.g. output.csv -> output_20260921_143022.csv
    """
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{timestamp}{ext}"


def write_file(path: str, content: str) -> str:
    """
    Write content to disk. If the path ends in .xlsx, treat content as
    CSV-formatted text and convert it into a real Excel file. Otherwise,
    write the content as plain text (works for .csv, .md, .txt).
    Never overwrites an existing file — auto-renames with a timestamp instead.
    """
    path = _make_unique_path(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if path.endswith(".xlsx"):
        try:
            df = pd.read_csv(io.StringIO(content))
        except Exception as e:
            return f"ERROR: could not parse content as CSV to convert to Excel: {e}"
        df.to_excel(path, index=False)
        return f"Wrote Excel file to {path} ({len(df)} rows)"

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote file to {path} ({len(content)} characters)"


# ---------- Registry: maps tool name -> (function, arg schema) ----------
# The Planner and Executor both use this to know what's available.

TOOL_REGISTRY = {
    "web_search": (web_search, WebSearchArgs),
    "run_python": (run_python, RunPythonArgs),
    "write_file": (write_file, WriteFileArgs),
}