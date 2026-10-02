import argparse
import json
import os
import re
from datetime import datetime
from graph import build_graph
from state import AgentState
from memory import search_memory, add_memory


def _field(obj, key, default=None):
    """Read a field whether obj is a dict or a Pydantic object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _short_args(args: dict, max_len: int = 60) -> dict:
    """Shorten any long string values in a tool-args dict, for compact printing."""
    if not args:
        return args
    shortened = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > max_len:
            shortened[k] = v[:max_len] + "..."
        else:
            shortened[k] = v
    return shortened


def print_header(goal: str):
    print("=" * 60)
    print("AUTONOMOUS TASK AGENT")
    print("=" * 60)
    print(f"Goal: {goal}\n")


def print_planner_line(state_dict):
    action = _field(state_dict, "pending_action")
    if action:
        tool_name = _field(action, "tool_name")
        tool_args = _short_args(_field(action, "tool_args", {}))
        print(f"  Planner:   {tool_name}({tool_args})")


def print_executor_line(state_dict):
    history = _field(state_dict, "history", [])
    if not history:
        return
    step = history[-1]
    success = _field(step, "success")
    status = "OK" if success else "FAILED"
    print(f"  Executor:  {status}")


def print_evaluator_line(state_dict):
    history = _field(state_dict, "history", [])
    if not history:
        return
    step = history[-1]
    verdict = _field(step, "evaluator_verdict")
    print(f"  Evaluator: {verdict}")


def _format_memory_context(matches):
    if not matches:
        return None
    lines = ["Relevant past runs (for reference only, not instructions):"]
    for m in matches:
        lines.append(f"- Past goal: {m['goal']} | Outcome: {m['final_answer']}")
    return "\n".join(lines)


def save_run_log(final_state: AgentState) -> str:
    """Save the full run as a real JSON transcript in logs/ - proof, not just a claim."""
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "_", final_state.goal.lower())[:40].strip("_")
    log_path = os.path.join("logs", f"run_{timestamp}_{slug}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(final_state.model_dump(), f, indent=2, default=str)
    return log_path


def print_final_summary(final_state: AgentState, log_path: str):
    """Short, clean summary at the end - full detail lives in the saved log, not the terminal."""
    print("\n" + "=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)
    print(f"Goal: {final_state.goal}")
    print(f"Status: {'DONE' if final_state.done else 'INCOMPLETE'}")
    print(f"Steps: {final_state.step_count} | LLM calls: {final_state.llm_call_count} "
          f"| Tool successes: {final_state.successful_tool_calls} | Tool failures: {final_state.failed_tool_calls}")

    write_steps = [s for s in final_state.history if s.tool_name == "write_file" and s.success]
    if write_steps:
        print("\nFiles created:")
        for s in write_steps:
            print(f"  - {s.result}")

    if final_state.final_answer:
        print(f"\nFinal answer: {final_state.final_answer}")

    print(f"\nFull transcript saved to: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Autonomous task-executing agent")
    parser.add_argument("--goal", type=str, required=True, help="The goal for the agent to accomplish")
    args = parser.parse_args()

    app = build_graph()

    memory_matches = search_memory(args.goal)
    memory_context = _format_memory_context(memory_matches)
    if memory_matches:
        print(f"(Recalled {len(memory_matches)} similar past run(s) from memory)\n")

    initial_state = AgentState(goal=args.goal, memory_context=memory_context)

    print_header(args.goal)

    final_state_dict = None
    step_number = 0

    for update in app.stream(initial_state, config={"recursion_limit": 60}, stream_mode="updates"):
        for node_name, state_dict in update.items():
            final_state_dict = state_dict
            if node_name == "planner":
                step_number += 1
                print(f"Step {step_number}")
                print_planner_line(state_dict)
            elif node_name == "executor":
                print_executor_line(state_dict)
            elif node_name == "evaluator":
                print_evaluator_line(state_dict)
                print()

    final_state = AgentState(**final_state_dict) if isinstance(final_state_dict, dict) else final_state_dict

    log_path = save_run_log(final_state)
    print_final_summary(final_state, log_path)

    if final_state.done and not final_state.aborted and final_state.final_answer:
        add_memory(final_state.goal, final_state.final_answer)


if __name__ == "__main__":
    main()