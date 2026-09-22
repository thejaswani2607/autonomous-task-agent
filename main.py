import argparse
from graph import build_graph
from state import AgentState


def _field(obj, key, default=None):
    """Read a field whether obj is a dict or a Pydantic object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def print_header(goal: str):
    print("=" * 60)
    print("AUTONOMOUS TASK AGENT")
    print("=" * 60)
    print(f"Goal: {goal}\n")


def print_planner_update(state_dict):
    action = _field(state_dict, "pending_action")
    if action:
        tool_name = _field(action, "tool_name")
        tool_args = _field(action, "tool_args")
        print(f"🧠 PLANNER decided: {tool_name}({tool_args})")


def print_executor_update(state_dict):
    history = _field(state_dict, "history", [])
    if not history:
        return
    step = history[-1]
    success = _field(step, "success")
    result = _field(step, "result", "")
    status = "✅ SUCCESS" if success else "❌ FAILED"
    print(f"⚙️  EXECUTOR ran it: {status}")
    print(f"    Result: {str(result)[:300]}")


def print_evaluator_update(state_dict):
    history = _field(state_dict, "history", [])
    if not history:
        return
    step = history[-1]
    verdict = _field(step, "evaluator_verdict")
    reasoning = _field(step, "evaluator_reasoning")
    if verdict:
        print(f"🔍 EVALUATOR verdict: {verdict.upper()}")
        print(f"    Reasoning: {reasoning}\n")


def main():
    parser = argparse.ArgumentParser(description="Autonomous task-executing agent")
    parser.add_argument("--goal", type=str, required=True, help="The goal for the agent to accomplish")
    args = parser.parse_args()

    app = build_graph()
    initial_state = AgentState(goal=args.goal)

    print_header(args.goal)

    final_state_dict = None

    for update in app.stream(initial_state, config={"recursion_limit": 60}, stream_mode="updates"):
        for node_name, state_dict in update.items():
            final_state_dict = state_dict
            if node_name == "planner":
                print_planner_update(state_dict)
            elif node_name == "executor":
                print_executor_update(state_dict)
            elif node_name == "evaluator":
                print_evaluator_update(state_dict)

    print("=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)

    final_state = AgentState(**final_state_dict) if isinstance(final_state_dict, dict) else final_state_dict
    print(final_state.pretty())

    # Cost/step summary — your improvement #2
    print("\n--- Run Summary ---")
    print(f"Total steps: {final_state.step_count}")
    print(f"LLM calls: {final_state.llm_call_count}")
    print(f"Successful tool calls: {final_state.successful_tool_calls}")
    print(f"Failed tool calls: {final_state.failed_tool_calls}")


if __name__ == "__main__":
    main()