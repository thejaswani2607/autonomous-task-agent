import json
import os
import glob

LOGS_DIR = "logs"


def load_all_runs():
    runs = []
    for path in glob.glob(os.path.join(LOGS_DIR, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                data["_file"] = os.path.basename(path)
                runs.append(data)
            except json.JSONDecodeError:
                print(f"Skipping unreadable file: {path}")
    return runs


def analyze(runs):
    total = len(runs)
    if total == 0:
        print("No log files found in logs/. Run the agent a few times first.")
        return

    completed = [r for r in runs if r.get("done") and not r.get("aborted")]
    aborted = [r for r in runs if r.get("aborted")]
    incomplete = [r for r in runs if not r.get("done")]
    user_declined = [r for r in runs if r.get("final_answer") and "declined" in r["final_answer"].lower()]

    genuinely_successful = [
        r for r in completed
        if r.get("final_answer") and "declined" not in r["final_answer"].lower()
        and "stopped" not in r["final_answer"].lower()
    ]

    avg_steps = sum(r.get("step_count", 0) for r in genuinely_successful) / len(genuinely_successful) if genuinely_successful else 0
    avg_llm_calls = sum(r.get("llm_call_count", 0) for r in genuinely_successful) / len(genuinely_successful) if genuinely_successful else 0

    total_tool_success = sum(r.get("successful_tool_calls", 0) for r in runs)
    total_tool_failure = sum(r.get("failed_tool_calls", 0) for r in runs)
    total_tool_calls = total_tool_success + total_tool_failure
    tool_success_rate = (total_tool_success / total_tool_calls * 100) if total_tool_calls else 0

    print("=" * 60)
    print("AGENT PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Total runs analyzed:         {total}")
    print(f"Genuinely successful runs:   {len(genuinely_successful)}")
    print(f"User-declined runs:          {len(user_declined)}")
    print(f"API-aborted runs:            {len(aborted)}")
    print(f"Incomplete/other:            {total - len(genuinely_successful) - len(user_declined) - len(aborted)}")
    print()
    if total > 0:
        completion_rate = len(genuinely_successful) / total * 100
        print(f"Task completion rate:        {completion_rate:.1f}%")
    print(f"Average steps (successful runs):     {avg_steps:.1f}")
    print(f"Average LLM calls (successful runs): {avg_llm_calls:.1f}")
    print()
    print(f"Total tool calls across all runs:    {total_tool_calls}")
    print(f"Tool-level success rate:             {tool_success_rate:.1f}%")
    print()
    print("Per-run breakdown:")
    for r in runs:
        status = "DONE" if r.get("done") and not r.get("aborted") else ("ABORTED" if r.get("aborted") else "INCOMPLETE")
        goal_preview = r.get("goal", "")[:55]
        print(f"  [{status:10}] {r.get('step_count', 0):2} steps | {goal_preview}")


if __name__ == "__main__":
    runs = load_all_runs()
    analyze(runs)