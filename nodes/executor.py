from state import AgentState, StepRecord
from tools import TOOL_REGISTRY


def execute_action(state: AgentState, tool_name: str, tool_args: dict, pre_approved: bool | None = None) -> AgentState:
    """
    Actually run the tool the Planner chose, record what happened,
    and return the updated state. Pauses for human approval before write_file
    via terminal input() in the CLI (pre_approved=None, the default).
    If pre_approved is True or False, that decision is used instead,
    skipping input() entirely - this is how the web app (app.py) drives
    approval through on-page buttons instead of the terminal.
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

    if tool_name == "write_file":
        if pre_approved is None:
            # CLI path: ask via terminal, exactly as before
            print("\n" + "=" * 50)
            print("The agent wants to write a file:")
            print(f"  Path: {tool_args.get('path')}")
            print(f"  Content preview:\n{tool_args.get('content', '')[:400]}")
            print("=" * 50)
            approval = input("Proceed? (y/n): ").strip().lower()
            approved = approval == "y"
        else:
            # Web path: approval already decided via on-page buttons
            approved = pre_approved

        if not approved:
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