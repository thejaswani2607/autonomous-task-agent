# Autonomous Task Agent

An AI agent that takes a plain-English goal, plans its own steps, uses real tools (web search, sandboxed code execution, file writing) to complete it, checks its own progress after every action, and keeps going — adapting and recovering from failures — until the goal is genuinely done or a safety limit is hit.

Example: you type `"Find 3 cheap project management tools and save the results as pm_tools_pricing.csv"`, and the agent searches the web for real pricing, decides when it has enough, writes a CSV, and stops — all on its own, with you only asked to approve the one moment it saves a file to your computer.

Available both as a command-line tool and as a deployed web app (built with Streamlit).

This project follows the **ReAct (Reason + Act) pattern**, a well-established design for autonomous agents, implemented as a cyclic graph using **LangGraph**.

**Live demo:** `<your Streamlit Community Cloud URL here>`

---

## 1. What this project actually does

Most chatbots answer one question and stop. This agent is different: it is given a **goal** and a small set of **tools**, and it repeatedly decides its own next move, carries it out, and judges whether it's done — in a loop — rather than following a fixed script.

Three specialized roles work in a cycle:

- **Planner** — looks at the goal and everything tried so far, and decides exactly ONE next action (which tool, with what input). It never plans the whole task up front, only the next single step.
- **Executor** — actually carries out that one action using a real tool, and records exactly what happened (success or failure, and the real output).
- **Evaluator** — looks at the result and judges: has the goal been fully achieved? Is more work needed? Did something fail badly enough that a different approach is needed?

If the Evaluator says the goal isn't done, control goes back to the Planner, which now has a full history of everything tried so far — so it doesn't blindly repeat failed actions. This repeats until the Evaluator confirms the goal is complete, or a hard safety limit (15 rounds) is reached.

All three roles are powered by the same underlying LLM (Google's Gemini) — they are not three separate AIs, just the same model given three different jobs and three different prompts.

### Architecture

User Goal (plain English)
|
v
+----------------+
| Planner Node | -> decides ONE next action (tool + arguments)
+--------+--------+
|
v
+----------------+
| Executor Node | -> runs the tool, records result/error
+--------+--------+
|
v
+----------------+
| Evaluator Node | -> judges: done? continue? stuck?
+--------+--------+
|
+------+-------+
| |
not done goal complete
(loop back or 15-step
to Planner) limit hit
| |
v v
(repeat) Final Answer


---

## 2. Two ways to use this agent

### Command-line (original interface)
Run directly from the terminal — streams each Planner/Executor/Evaluator step live, pauses in-terminal for `y`/`n` approval before writing any file.

### Web app (Streamlit)
A browser-based interface with the same underlying agent logic — type a goal, watch it work step by step, click Approve/Reject buttons instead of typing, and download the result file directly from the page. Deployed for free on **Streamlit Community Cloud**, connected directly to this GitHub repository.

Both interfaces share the exact same core logic (`tools.py`, `nodes/`, `graph.py`, `state.py`, `memory.py`, `llm_utils.py`) — only the way you interact with the agent differs.

---

## 3. The tools the agent can use

The agent's abilities are entirely defined by these three tools — if a task can't be done with some combination of these, the agent cannot do it.

| Tool | What it does |
|---|---|
| `web_search` | Searches the web (via the Tavily API) and returns real, current results — titles, snippets, and source links. |
| `run_python` | Runs Python code in a fully isolated, throwaway sandbox (a separate subprocess, in its own temporary folder, with a timeout). Used for genuine computation — math, sorting, merging data — not for saving real files. |
| `write_file` | Saves content to a real file on disk (CSV or Excel/.xlsx). This is the only tool that produces a real, persistent output, and it is the one action that requires the user's explicit approval before it runs. |

### What this agent is good at

- Finding information online and compiling it into a file (price comparisons, lists, research summaries)
- Light data processing before saving (combining, deduplicating, simple calculations)
- Tasks with a clear, checkable "done" condition (e.g. "find 3 things and save them")

### What this agent cannot do (by design, not oversight)

- Anything requiring login or an existing account (checking email, posting to social media)
- Clicking buttons or navigating websites like a browser-automation tool would
- Understanding images, audio, or video
- Taking irreversible real-world actions beyond saving a local file (no sending messages, no purchases)
- Very open-ended, subjective tasks with no clear finish line (the Evaluator needs something concrete to check against)

---

## 4. Key design decisions and safety features

These were deliberate choices made while building this, not accidents — each one addresses a specific real problem encountered during development and testing.

**Sandboxed code execution.** `run_python` never runs inside the main program. Code is written to a temporary file and run as a separate subprocess, inside its own throwaway temporary working directory, with a timeout. The AI-generated code cannot crash the main agent, and cannot write real files to the actual project folder even if it tries to — only `write_file` can produce persistent output. The temp folder and everything in it is deleted immediately after.

**Human-in-the-loop approval.** Before `write_file` ever touches the real filesystem, the agent pauses, shows exactly what it's about to write, and waits for explicit confirmation — typed `y`/`n` on the CLI, or an Approve/Reject button click on the web app. Nothing irreversible happens without it. **Rejecting immediately stops the run** — it does not retry or ask again.

**Never overwrites existing files.** If the requested output filename already exists (e.g. from a previous run), the agent automatically appends a timestamp instead of overwriting it (e.g. `output.csv` becomes `output_20261002_143022.csv`).

**The Evaluator's judgment is not blindly trusted.** Early testing showed that trusting the LLM's own sense of "I'm done" can be wrong — for example, declaring a goal complete after finding information but before actually saving a file, or claiming "3 items found" after only 2 real searches. Two code-level safety checks override the Evaluator's verdict regardless of what it claims:
1. If the goal implies a file must be saved, "done" is rejected unless a `write_file` call has actually succeeded in the history.
2. If the goal specifies a quantity (e.g. "3 recipes", "5 tools"), "done" is rejected unless at least that many successful `web_search` calls have occurred.

These are heuristics, not perfect guarantees, triggered only when the goal's wording implies a checkable requirement — a goal with no number and no file-saving language correctly skips these checks and relies on the Evaluator's own judgment (see the Performance Metrics section for a real example of this).

**Retry with backoff, and automatic model fallback.** The agent automatically retries temporary API errors with increasing delays, and falls back to a different model if one keeps failing. If every model and every retry genuinely fails, the run ends with a clear explanation instead of crashing.

**A hard step limit.** The loop is capped at 15 rounds regardless of what the Evaluator says.

**Memory across runs.** A small local memory (via FAISS and `sentence-transformers`, free and fully local) recalls similar past goals and gives that as reference context to the Planner — never as a replacement for re-verifying with a real search.

**Every run is logged.** The full details of every run are saved as a real JSON file in `logs/` — proof, not a claim.

**Cost/step tracking.** Every run reports total steps, LLM calls, and tool success/failure counts.

---

## 5. Real performance metrics

The project includes `analyze_logs.py`, a script that reads every saved run transcript in `logs/` and computes aggregate statistics — not an estimate, a number computed directly from real, inspectable evidence.

**Results across 10 real test runs:**

| Metric | Result |
|---|---|
| Total runs analyzed | 10 |
| Genuinely successful runs | 9 |
| Task completion rate | 90.0% |
| Average steps (successful runs) | 4.2 |
| Average LLM calls (successful runs) | 8.4 |
| Tool-level success rate | 97.5% |

The one non-success in this sample was a deliberate test of the reject-approval safety path (rejecting a file save), not an agent failure — the run correctly stopped immediately rather than retrying, which is the intended behavior.

Run it yourself anytime to get fresh numbers as you test more goals:
```bash
python analyze_logs.py
```

---

## 6. Known limitations (stated honestly)

No system built on an LLM's judgment is perfect, and it's more honest to state the real limits than to pretend they don't exist:

- The "enough searches were done" safety check counts *successful search calls*, not verified *distinct* items — a Planner could technically run several overlapping searches and this check would not catch that.
- Safety checks are keyword-triggered (looking for numbers and file-related words in the goal text) — a goal phrased without these cues will rely entirely on the Evaluator's own judgment, without the code-level backstop.
- The Evaluator's `final_answer` text is instructed to only reference items it can trace to real search evidence, but as an LLM-generated summary, it is not under a hard code-level guarantee the way the two safety checks above are.
- `run_python`'s sandbox does not have access to third-party libraries like `pandas` (only Python's standard library).
- Efficiency (how many steps a run takes) depends on the Planner's own judgment — significantly improved by prompt tuning during development, but not perfectly optimal on every run.
- Memory recall is based on semantic similarity of the goal text, not deep understanding of whether a recalled past run is still relevant or current.
- **The hosted web app's filesystem is not guaranteed to persist** across Streamlit Community Cloud redeploys or sleep/wake cycles — `logs/` and `memory_store/` written during a hosted session may not reliably survive a redeploy. A production version would use a proper persistent store (a hosted database) instead of local files for these.

---

## 7. Project structure

autonomous-task-agent/
├── nodes/
│ ├── init.py
│ ├── planner.py # decides the next single action
│ ├── executor.py # runs the chosen tool, handles approval for write_file
│ └── evaluator.py # judges progress, enforces safety checks
├── tools.py # web_search, run_python (sandboxed), write_file
├── state.py # AgentState schema - the shared memory of one run
├── graph.py # wires the three nodes into the LangGraph loop
├── llm_utils.py # retry-with-backoff and model fallback logic
├── memory.py # FAISS-based memory across runs
├── main.py # CLI entrypoint
├── app.py # Streamlit web app entrypoint
├── analyze_logs.py # computes aggregate performance stats from logs/
├── .streamlit/
│ └── config.toml # web app color theme
├── requirements.txt
├── .env.example
├── .gitignore
├── logs/ # real saved transcripts of every run (JSON)
├── memory_store/ # local FAISS index (not committed to git)
└── README.md


---

## 8. How to run this project

### Prerequisites

- Python 3.11 (built and tested on 3.11.9; very new Python versions may lack pre-built wheels for some dependencies)
- A free Google AI Studio account, for a Gemini API key: https://aistudio.google.com
- A free Tavily account, for a web search API key: https://tavily.com

### Setup

```bash
# Clone the repository
git clone https://github.com/thejaswani2607/autonomous-task-agent.git
cd autonomous-task-agent

# Create and activate a virtual environment (Python 3.11 specifically)
py -3.11 -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### API keys

Copy `.env.example` to a new file named `.env`, and fill in your real keys:

GOOGLE_API_KEY=your_real_gemini_api_key
TAVILY_API_KEY=your_real_tavily_api_key

`.env` is already excluded from version control via `.gitignore` — never commit your real keys.

### Option A — Run via command line

```bash
python main.py --goal "Find 3 cheap project management tools and save the results as pm_tools_pricing.csv"
```

The agent prints a short line for each step (Planner's decision, Executor's result, Evaluator's verdict), pauses in-terminal to ask `y`/`n` before saving any file, and finishes with a summary. The full detailed transcript is always saved to `logs/`.

### Option B — Run the web app locally

```bash
streamlit run app.py
```

This opens a browser tab at `http://localhost:8501`. Type a goal, click "Run Agent," and watch it work. When it wants to save a file, Approve/Reject buttons appear on the page instead of a terminal prompt. A download button appears once the run completes.

### Option C — Use the deployed web app

No installation needed — open the live link: `<your Streamlit Community Cloud URL here>`

*(Note: this requires the deployer's own API keys to be configured as Streamlit secrets; a visitor does not need their own keys to use the hosted demo.)*

### Checking real performance stats

```bash
python analyze_logs.py
```

Reads every file in `logs/` and prints an aggregate report: completion rate, average steps, average LLM calls, and tool-level success rate.

### Example goals to try

```bash
python main.py --goal "Find 3 cheap project management tools and save the results as pm_tools_pricing.csv"

python main.py --goal "Find the current prices of 3 popular coffee makers, calculate their average price, and save a summary as coffee_makers.csv"

python main.py --goal "Search for 3 recipes for chicken tikka masala, extract their ingredient lists, and save a combined shopping list as tikka_shopping_list.csv"

python main.py --goal "Find the 3 cheapest wireless earbuds under $100 and save the results as earbuds.csv"

python main.py --goal "Find the 3 tallest buildings in the world and save their names, heights, and locations as tallest_buildings.csv"

python main.py --goal "Find 3 cheap meal-kit delivery services and save their pricing as meal_kits.xlsx"
```

(Replace `python main.py --goal "..."` with the web app's text box for the same goals there.)

A good goal for this agent generally has three properties: the needed information is findable via a normal web search (not behind a login), there's a clear, checkable "done" condition, and you could personally verify the output file is correct afterward.

---

## 9. Deployment

The web app is deployed on **Streamlit Community Cloud**, a free hosting service that connects directly to this GitHub repository.

- Pushing to GitHub is effectively the deployment step — Streamlit Community Cloud automatically redeploys on new commits to the connected branch.
- API keys (`GOOGLE_API_KEY`, `TAVILY_API_KEY`) are configured as **secrets** in the Streamlit dashboard, not committed to the repository — same principle as the local `.env` file.
- See the Known Limitations section above regarding the hosted filesystem's persistence.

---

## 10. Tech stack

- **Orchestration:** LangGraph (cyclic graph implementation of the ReAct pattern)
- **LLM:** Google Gemini, via the `google-genai` SDK, with native function-calling for tool selection (no manual text parsing)
- **Web search:** Tavily API
- **Memory:** FAISS (vector similarity search) + `sentence-transformers` (local, free text embeddings)
- **Web interface:** Streamlit, deployed on Streamlit Community Cloud
- **Data handling:** pandas, openpyxl (for Excel output)
- **Schema validation:** Pydantic

---

## 11. License

This project is open for personal and educational use.