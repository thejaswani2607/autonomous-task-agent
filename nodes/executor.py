from state import AgentState, StepRecord
from tools import TOOL_REGISTRY


def execute_action(state: AgentState, tool_name: str, tool_args: dict) -> AgentState:
    """
    Actually run the tool the Planner chose, record what happened,
    and return the updated state. Pauses for human approval before write_file.
    """
    state.step_count += 1

    if tool_name not in TOOL_REGISTRY:
        record = StepRecord(
            step_number=state.step_count,
            tool_name=tool_name or "unknown",
            tool_args=tool_args,
            result=f"ERROR: unknown tool '{tool_name}'",
            success=False,
        )
        state.history.append(record)
        state.failed_tool_calls += 1
        return state

    func, _ = TOOL_REGISTRY[tool_name]

    # Human-in-the-loop checkpoint: pause before any real filesystem write
    if tool_name == "write_file":
        print("\n" + "=" * 50)
        print("The agent wants to write a file:")
        print(f"  Path: {tool_args.get('path')}")
        print(f"  Content preview:\n{tool_args.get('content', '')[:400]}")
        print("=" * 50)
        approval = input("Proceed? (y/n): ").strip().lower()
        if approval != "y":
            record = StepRecord(
                step_number=state.step_count,
                tool_name=tool_name,
                tool_args=tool_args,
                result="User declined this action.",
                success=False,
            )
            state.history.append(record)
            state.failed_tool_calls += 1
            return state

    try:
        result = func(**tool_args)
        success = not str(result).startswith("ERROR")
    except Exception as e:
        result = f"ERROR: {e}"
        success = False

    record = StepRecord(
        step_number=state.step_count,
        tool_name=tool_name,
        tool_args=tool_args,
        result=str(result),
        success=success,
    )
    state.history.append(record)

    if success:
        state.successful_tool_calls += 1
    else:
        state.failed_tool_calls += 1

    return state