# Autonomous Task Agent

An AI agent that takes a plain-English goal, plans its own steps, uses real tools (web search, sandboxed code execution, file writing) to complete it, checks its own progress after every action, and keeps going, adapting and recovering from failures, until the goal is genuinely done or a safety limit is hit.

Example: you type `"Find 3 cheap project management tools and save the results as pm_tools_pricing.csv"`, and the agent searches the web for real pricing, decides when it has enough, asks your permission, writes a CSV, and stops. You are only asked to approve the one moment it saves a file to your computer.

It works both as a command-line tool and as a deployed web app (built with Streamlit).

This project follows the **ReAct (Reason + Act) pattern**, a well-established design for autonomous agents, implemented as a cyclic graph using **LangGraph**.

**Live demo:** `<your Streamlit Community Cloud URL here>`

---

## 1. What this project actually does

Most chatbots answer one question and stop. This agent is different: it is given a **goal** and a small set of **tools**, and it repeatedly decides its own next move, carries it out, and judges whether it is done, in a loop, rather than following a fixed script.

Three specialized roles work in a cycle:

- **Planner**: looks at the goal and everything tried so far, and decides exactly ONE next action (which tool, with what input). It never plans the whole task up front, only the next single step.
- **Executor**: carries out that one action using a real tool, and records exactly what happened (success or failure, and the real output).
- **Evaluator**: looks at the result and judges: has the goal been fully achieved? Is more work needed? Did something fail badly enough that a different approach is needed?

If the Evaluator says the goal is not done, control goes back to the Planner, which now has a full history of everything tried so far, so it does not blindly repeat failed actions. This repeats until the Evaluator confirms the goal is complete, or a hard safety limit (15 rounds) is reached.

All three roles are powered by the same underlying LLM (Google's Gemini). They are not three separate AIs, just the same model given three different jobs and three different prompts.

### Architecture

```
User Goal (plain English)
        |
        v
 +----------------+
 |  Planner Node   |  -> decides ONE next action (tool + arguments)
 +--------+--------+
          |
          v
 +----------------+
 | Executor Node   |  -> runs the tool, records result/error
 +--------+--------+      (a declined file save ends the run here)
          |
          v
 +----------------+
 | Evaluator Node  |  -> judges: done? continue? stuck?
 +--------+--------+
          |
   +------+-------+
   |              |
 not done      goal complete
 (loop back    or 15-step
  to Planner)   limit hit
   |              |
   v              v
 (repeat)     Final Answer
```

---

## 2. Two ways to use this agent

### Command line
Run directly from the terminal. It prints one short line per step (Planner, Executor, Evaluator) and pauses in the terminal for `y`/`n` approval before writing any file.

### Web app (Streamlit)
A browser-based interface running the same agent logic: type a goal, watch it work step by step, click Approve or Reject instead of typing, and download the result file from the page. It is deployed for free on **Streamlit Community Cloud**, connected directly to this GitHub repository.

Both interfaces share the same core logic (`tools.py`, `nodes/`, `graph.py`, `state.py`, `memory.py`, `llm_utils.py`, `checks.py`). Only the way you interact with the agent differs.

---

## 3. The tools the agent can use

The agent's abilities are entirely defined by these three tools. If a task cannot be done with some combination of them, the agent cannot do it.

| Tool | What it does |
|---|---|
| `web_search` | Searches the web (via the Tavily API) and returns real, current results: titles, snippets, and source links. |
| `run_python` | Runs Python code in an isolated, throwaway sandbox (a separate subprocess, in its own temporary folder, with a timeout). Used for genuine computation such as math, sorting, or merging data, not for saving real files. |
| `write_file` | Saves content to a real file on disk (CSV or Excel/.xlsx). This is the only tool that produces a persistent output. It only runs if the goal asked for a file, only if a CSV/Excel table is well-formed, and only after the user approves it. |

### What this agent is good at

- Finding information online and compiling it into a file (price comparisons, lists, research summaries)
- Answering research questions directly (for example, "find excellent budget hotels in Jaipur"), with no file at all
- Light data processing before saving (combining, deduplicating, simple calculations)
- Tasks with a clear, checkable "done" condition (for example, "find 3 things and save them")

### What this agent cannot do (by design)

- Anything requiring login or an existing account (checking email, posting to social media)
- Clicking buttons or navigating websites like a browser-automation tool would
- Understanding images, audio, or video
- Taking irreversible real-world actions beyond saving a local file (no sending messages, no purchases)
- Very open-ended, subjective tasks with no clear finish line (the Evaluator needs something concrete to check against)

### How to ask for a file

A file is only ever saved if your goal **asks for one**. Use a word like `save`, `csv`, `excel`, `xlsx`, `spreadsheet` or `file`, or a filename like `report.json`. If your goal does not mention any of these, the agent answers in text and never writes a file. You do not need to worry about commas inside values (for example "iOS, Android, Web"): the agent is told to wrap them in quotes, and a badly formed table is refused and retried automatically.

---

## 4. Key design decisions and safety features

These were deliberate choices, each addressing a specific problem found during development and testing.

**Sandboxed code execution.** `run_python` never runs inside the main program. Code is written to a temporary file and run as a separate subprocess, inside its own throwaway working directory, with a timeout. A crash or infinite loop in AI-generated code cannot take down the agent, and ordinary file writes land in the temporary folder, which is deleted afterward, not in the project. This is isolation from crashes and stray file writes, **not a hardened security sandbox**: it does not block network access. A production deployment would use containers.

**A file is only saved if the goal asked for it, and only with your approval.** The Executor checks the goal's wording before any save. If no file was requested, the save is blocked in code: no prompt, nothing written. If a file was requested, the agent shows exactly what it is about to write and waits for explicit confirmation, typed `y`/`n` on the command line or an Approve/Reject click in the web app.

**Tables are checked before anything is saved.** CSV and Excel content is read with Python's `csv` module, which understands quotes. If any row has a different number of values from the header (the usual cause is a value containing a comma that is not wrapped in double quotes), the save is refused, before you are asked to approve it, with a message telling the agent exactly how to fix it. The agent then retries, which costs one extra step. This replaced an earlier behavior where such a table was written with shifted columns and lost data. Excel files are built from the checked rows. Columns made only of plain numbers become real numbers, while IDs with leading zeros, phone numbers and prices with currency symbols stay text so nothing is damaged. CSV files are saved with a UTF-8 marker so Excel shows symbols like the rupee sign correctly.

**Declining stops the run.** If you decline a save, the run ends immediately. The agent does not retry or ask again. The transcript is still logged, but the run is not counted as a success and is not stored in memory.

**Never overwrites existing files.** If the requested filename already exists, a timestamp is added instead (for example `output.csv` becomes `output_20261002_143022.csv`).

**The Evaluator's judgment is not blindly trusted.** Early testing showed that an LLM's own sense of "I'm done" can be wrong, for example declaring a goal complete before a file was saved, or claiming "3 items found" after only 2 real searches. Two code-level checks override the Evaluator's verdict:
1. If the goal asks for a file, "done" is rejected unless a `write_file` call has actually succeeded.
2. If the goal specifies a count (like "3 recipes"), "done" is rejected unless at least that many successful `web_search` calls have occurred.

Both checks read the goal's wording using whole-word matching, so "profile" does not count as "file", and a number only counts as an item count if it is not a unit or limit ("5 minutes", "under 5", "2 cups", and "2-in-1" are ignored). These are heuristics, not perfect guarantees (see Known limitations).

**Retry with backoff, and automatic model fallback.** Temporary API errors (an overloaded model, a dropped connection) are retried with increasing delays, and the agent falls back to a different model if one keeps failing. If every model and retry fails, the run ends with a clear explanation instead of crashing, in both the command line and the web app.

**A hard step limit.** The loop is capped at 15 rounds regardless of what the Evaluator says.

**Memory across runs, that repairs itself.** A small local memory (FAISS plus `sentence-transformers`, free and fully local) recalls similar past goals and gives them to the Planner as reference context, never as a replacement for a real search. Only genuinely successful runs are stored. `metadata.json` is the source of truth, and the FAISS lookup is rebuilt from it automatically if the two files ever disagree.

**Every run is logged.** The full details of every run are saved as a JSON file in `logs/`: proof, not just a claim.

**Cost and step tracking.** Every run reports total steps, LLM calls, and tool success/failure counts.

---

## 5. Real performance metrics

`analyze_logs.py` reads every saved run transcript in `logs/` and computes aggregate statistics. These are numbers computed from real, inspectable evidence, not estimates.

**Initial baseline: the first 10 real test runs** (made before several later fixes):

| Metric | Result |
|---|---|
| Total runs analyzed | 10 |
| Genuinely successful runs | 9 |
| Task completion rate | 90.0% |
| Average steps (successful runs) | 4.2 |
| Average LLM calls (successful runs) | 8.4 |
| Tool-level success rate | 97.5% |

The one non-success in that sample was a deliberate test of the reject-approval path, not an agent failure. The run stopped immediately instead of retrying, which is the intended behavior.

`logs/` now contains more runs than this baseline, so the current script output will differ. Run it yourself for fresh numbers:

```bash
python analyze_logs.py
```

**What "successful" means here:** the run reached a verified "done" state. It was not aborted (Gemini was available), not stopped by the 15-step limit, and not declined by the user. It is **not** a human grade of answer quality. Making the Evaluator stricter about specific answers also tends to cost extra steps, which you will see in the average.

---

## 6. Known limitations (stated honestly)

No system built on an LLM's judgment is perfect, and it is more honest to state the real limits.

- The count safety check counts *successful search calls*, not verified *distinct* items. A Planner could run several overlapping searches and pass it.
- A file can be saved before the count check is satisfied. The Planner counts the items it can see in the search results, while the check counts searches, so after saving, the agent may do a few more searches only to satisfy the count rule. The saved file is not affected (it is written once and never rewritten), but the run takes extra steps. A better design would block the save until enough searches are done.
- In Excel files, formatted numbers such as `$19.99` or `1,200` stay as text. Only plain numbers are converted, on purpose.
- The safety checks read the goal's wording. Spelled-out numbers ("three") are not detected, and only the first real number in a goal is used. A goal that wants a file must use a file word (see "How to ask for a file").
- The Evaluator's `final_answer` is instructed to name only items traceable to real search results, and to name one specific pick for "best option" goals, but this is a prompt rule, not a code-level guarantee.
- `run_python`'s sandbox only has Python's standard library (no `pandas`), and is not a hardened security boundary.
- Efficiency (steps per run) depends on the Planner's and Evaluator's judgment. Prompt tuning improved it, but it is not optimal on every run, and stricter answer requirements can add steps.
- Memory recall is based on semantic similarity of the goal text, not on whether a recalled past run is still current.
- There is no automated test suite. Verification was done through real runs, with every run's transcript kept in `logs/`.
- **The hosted web app's filesystem is not guaranteed to persist** across Streamlit Community Cloud redeploys or sleep/wake cycles. `logs/` and `memory_store/` written during a hosted session may not survive. A production version would use a hosted database.

---

## 7. Project structure

```
autonomous-task-agent/
├── nodes/
│   ├── __init__.py
│   ├── planner.py        # decides the next single action
│   ├── executor.py        # runs the chosen tool; blocks unrequested saves, handles approval
│   └── evaluator.py        # judges progress, enforces the two safety checks
├── tools.py                 # web_search, run_python (sandboxed), write_file
├── state.py                  # AgentState schema, the shared memory of one run
├── graph.py                   # wires the three nodes into the LangGraph loop
├── llm_utils.py                # retry-with-backoff and model fallback
├── memory.py                    # FAISS memory across runs (self-healing)
├── checks.py                     # goal-reading helpers shared by evaluator, executor, main, app
├── main.py                        # command-line entrypoint
├── app.py                          # Streamlit web app entrypoint
├── analyze_logs.py                  # computes aggregate performance stats from logs/
├── .streamlit/
│   └── config.toml                   # web app color theme
├── requirements.txt
├── .env.example
├── .gitignore
├── logs/                              # saved transcripts of every run (JSON)
├── memory_store/                       # local memory files (not committed to git)
└── README.md
```

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

```
GOOGLE_API_KEY=your_real_gemini_api_key
TAVILY_API_KEY=your_real_tavily_api_key
```

`.env` is excluded from version control via `.gitignore`. Never commit your real keys.

### Option A: Run via command line

```bash
python main.py --goal "Find 3 cheap project management tools and save the results as pm_tools_pricing.csv"
```

You will see one short line per step, a `y`/`n` prompt before any file is saved, and a summary at the end. A run that ends without completing its goal shows `Status: STOPPED`. The full transcript is always saved to `logs/`.

### Option B: Run the web app locally

```bash
streamlit run app.py
```

This opens `http://localhost:8501`. Type a goal and click "Run Agent". If your goal asks for a file, Approve/Reject buttons appear when the agent wants to save it, and a download button appears when the run completes.

### Option C: Use the deployed web app

Open the live link: `<your Streamlit Community Cloud URL here>`

### Checking performance stats

```bash
python analyze_logs.py
```

### Example goals to try

**Goals that produce a file:**

```bash
python main.py --goal "Find 3 cheap electric kettles and save their prices as kettles.csv"

python main.py --goal "Find the current prices of 3 popular coffee makers, calculate their average price, and save a summary as coffee_makers.csv"

python main.py --goal "Search for 3 recipes for chicken tikka masala, extract their ingredient lists, and save a combined shopping list as tikka_shopping_list.csv"

python main.py --goal "Find the 3 tallest buildings in the world and save their names, heights, and locations as tallest_buildings.csv"

python main.py --goal "Find 3 cheap meal-kit delivery services and save their pricing as meal_kits.xlsx"
```

**Goals that just answer (no file is created or requested):**

```bash
python main.py --goal "Find excellent budget hotels in Jaipur"

python main.py --goal "Give me a pasta recipe that takes under 15 minutes"

python main.py --goal "Find the flavor profile of a good espresso"

python main.py --goal "Find a 2-in-1 laptop under 60000 rupees and tell me the best option"
```

A good goal has three properties: the information is findable with a normal web search (not behind a login), there is a clear, checkable "done" condition, and you could personally verify the result afterward.

### Troubleshooting

- **`ModuleNotFoundError` for a package you installed:** your virtual environment is not active. Look for `(venv)` at the start of your prompt and re-run the activate command.
- **`503 UNAVAILABLE` from Gemini:** this is the free tier being busy. The agent retries and falls back to other models automatically. If it still ends with an "unavailable" message, run it again in a few minutes.
- **No file was created:** your goal did not ask for one. Add `save as something.csv` (or a similar file word) to the goal.
- **A step shows `FAILED` with "not saved - row N has X values but the header has Y":** the agent sent a badly formed table (usually a value with a comma and no quotes). Nothing was written and you were not asked to approve it. The agent reads the message and retries automatically, so no action is needed.
- **Memory seems wrong or you want a clean slate:** delete the `memory_store/` folder. It rebuilds itself as you use the agent. The memory also repairs itself automatically if its two files drift out of sync.
- **PowerShell and `$` signs:** inside double quotes, PowerShell can treat `$100` as a variable and drop it from your goal. Write "100 dollars" instead, or wrap the whole goal in single quotes.

---

## 9. Deployment

The web app is deployed on **Streamlit Community Cloud**, a free hosting service connected directly to this GitHub repository.

- Pushing to GitHub is effectively the deployment step. Streamlit Community Cloud redeploys automatically on new commits to the connected branch.
- API keys (`GOOGLE_API_KEY`, `TAVILY_API_KEY`) are configured as **secrets** in the Streamlit dashboard, not committed to the repository.
- See Known limitations regarding the hosted filesystem's persistence.

---

## 10. Tech stack

- **Orchestration:** LangGraph (cyclic graph implementation of the ReAct pattern)
- **LLM:** Google Gemini, via the `google-genai` SDK, with native function-calling for tool selection (no manual text parsing)
- **Web search:** Tavily API
- **Memory:** FAISS (vector similarity search) and `sentence-transformers` (local, free text embeddings)
- **Web interface:** Streamlit, deployed on Streamlit Community Cloud
- **Data handling:** pandas, openpyxl (for Excel output)
- **Schema validation:** Pydantic

---

## 11. License

This project is open for personal and educational use.