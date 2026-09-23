import subprocess
import tempfile
import os
import io
import re
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

def _clean_snippet(text: str) -> str:
    """
    Strip common webpage-scraping noise from raw search content:
    markdown headers, star-rating symbols, image placeholder labels,
    and excessive whitespace/line breaks. Keeps the real sentences intact.
    """
    if not text:
        return text

    # Remove markdown heading markers like ###, ######
    text = re.sub(r"#{1,6}\s*", "", text)

    # Remove star-rating symbols
    text = text.replace("⭐", "")

    # Remove common image/UI placeholder tokens
    text = re.sub(r"\b(pros-image|cons-image)\b", "", text)

    # Collapse multiple newlines/spaces into a single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def web_search(query: str) -> str:
    """Search the web and return a readable, cleaned summary of top results."""
    response = tavily_client.search(query=query, max_results=5)
    results = response.get("results", [])
    if not results:
        return "No results found."

    lines = []
    for r in results:
        cleaned = _clean_snippet(r.get("content", ""))[:250]
        lines.append(f"- {r.get('title')}: {cleaned} (source: {r.get('url')})")
    return "\n".join(lines)


def run_python(code: str, timeout: int = 10) -> str:
    """
    Run Python code in an isolated subprocess with a timeout, inside a
    throwaway temporary directory. This means the code cannot crash our
    main program AND cannot write real files to the actual project folder,
    even if it tries to (e.g. via open()) - any file writes land in a
    temp folder that gets deleted right after. Only write_file is allowed
    to produce real, persistent output, since that's the one path that
    goes through human approval.
    """
    with tempfile.TemporaryDirectory() as sandbox_dir:
        script_path = os.path.join(sandbox_dir, "script.py")
        with open(script_path, "w") as f:
            f.write(code)

        try:
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=sandbox_dir,
            )
            if result.returncode != 0:
                return f"ERROR (exit code {result.returncode}): {result.stderr.strip()}"
            return result.stdout.strip() if result.stdout.strip() else "(code ran successfully, no output printed)"
        except subprocess.TimeoutExpired:
            return f"ERROR: code timed out after {timeout} seconds"


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
    This is the ONLY tool that produces real, persistent output.
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