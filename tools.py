import subprocess
import tempfile
import os
import io
import re
import csv
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

    text = re.sub(r"#{1,6}\s*", "", text)
    text = text.replace("⭐", "")
    text = re.sub(r"\b(pros-image|cons-image)\b", "", text)
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
    throwaway temporary directory. A crash or infinite loop cannot take down
    the main program, and ordinary file writes land in the temporary folder
    (which is deleted afterward), not in the project.
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
    """
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{timestamp}{ext}"


# ---------- Table (CSV / Excel) handling ----------
#
# The agent writes tables as CSV text. If a value contains a comma and is not wrapped
# in double quotes (for example  Duolingo,Freemium,iOS, Android, Web ), that row gets
# extra values, and a careless converter silently shifts the columns and loses data.
# So tables are read with Python's csv module (which understands quotes), checked
# BEFORE anything is saved or shown for approval, and rebuilt explicitly.

TABLE_EXTENSIONS = (".csv", ".xlsx")


def _parse_table(content):
    """
    Read CSV text strictly. Returns (rows, None) on success, where rows[0] is the
    header and every row has the same number of cells, or (None, problem) with a
    short explanation the agent can act on.
    """
    text = str(content or "").lstrip("\ufeff")
    try:
        raw_rows = list(csv.reader(io.StringIO(text), skipinitialspace=True, strict=True))
    except csv.Error as e:
        return None, (
            f"the CSV text is malformed ({e}). "
            f"Every opening double quote needs a matching closing quote."
        )

    # strip whitespace around every cell and ignore completely blank lines
    rows = [[cell.strip() for cell in row] for row in raw_rows if any(cell.strip() for cell in row)]

    if len(rows) < 2:
        return None, "the table needs a header row and at least one data row."

    header = rows[0]
    if any(name == "" for name in header):
        return None, "the header row has an empty column name."

    for number, row in enumerate(rows[1:], start=1):
        if len(row) != len(header):
            return None, (
                f"row {number} has {len(row)} values but the header has {len(header)}. "
                f'Wrap any value that contains a comma in double quotes, e.g. Acme,Free,"iOS, Android, Web". '
                f"Row was: {','.join(row)[:80]}"
            )

    return rows, None


def check_file_content(path, content):
    """
    Used BEFORE asking the user to approve a save. Returns a short description of what
    is wrong with the content, or None if it is fine. Only .csv and .xlsx files are
    checked; other file types (.md, .txt, ...) are always accepted.
    """
    if not isinstance(path, str) or not path.lower().endswith(TABLE_EXTENSIONS):
        return None
    _, problem = _parse_table(content)
    return problem


def _looks_numeric(value: str) -> bool:
    """Plain decimal numbers only (no leading zeros, no '+', no exponent), so IDs and phone numbers stay text."""
    if not re.fullmatch(r"-?\d+(\.\d+)?", value):
        return False
    return not re.fullmatch(r"-?0\d+(\.\d+)?", value)


def _rows_to_dataframe(rows):
    """Build the Excel table from already-validated rows. Columns that are entirely plain numbers become numeric."""
    df = pd.DataFrame(rows[1:], columns=rows[0])
    for i in range(df.shape[1]):
        column = df.iloc[:, i]
        filled = [v for v in column if v != ""]
        if filled and all(_looks_numeric(v) for v in filled):
            df.isetitem(i, pd.to_numeric(column.where(column != "", None)))
    return df


def write_file(path: str, content: str) -> str:
    """
    Write content to disk.

    .csv / .xlsx: the content must be a valid CSV table (see _parse_table). A malformed
    table is refused with an explanation instead of being saved with wrong data.
    .xlsx files are built explicitly from the validated rows; .csv files are rewritten
    with correct quoting and a UTF-8 marker so Excel shows symbols like the rupee sign
    correctly. Other file types (.md, .txt, ...) are written as plain text.

    Never overwrites an existing file - auto-renames with a timestamp instead.
    """
    if isinstance(path, str) and path.lower().endswith(TABLE_EXTENSIONS):
        rows, problem = _parse_table(content)
        if problem:
            return f"ERROR: not saved - {problem}"

        path = _make_unique_path(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if path.lower().endswith(".xlsx"):
            df = _rows_to_dataframe(rows)
            df.to_excel(path, index=False)
            return f"Wrote Excel file to {path} ({len(df)} rows)"

        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="\n").writerows(rows)
        text = buffer.getvalue()
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            f.write(text)
        return f"Wrote file to {path} ({len(text)} characters)"

    path = _make_unique_path(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
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