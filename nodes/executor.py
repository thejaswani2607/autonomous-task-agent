from state import AgentState, StepRecord
from tools import TOOL_REGISTRY, check_file_content
from checks import goal_requires_file_output

DECLINED_MESSAGE = "Run stopped: you declined to save the file, so the agent did not continue."


def execute_action(state: AgentState, tool_name: str, tool_args: dict, pre_approved: bool | None = None) -> AgentState:
    """
    Actually run the tool the Planner chose, record what happened,
    and return the updated state.

    write_file has three protections:
      1. It is BLOCKED if the goal never asked for a saved file (so the agent
         cannot create files nobody requested). No approval prompt is shown.
      2. For .csv / .xlsx files, the table text must be valid (every row has as many
         values as the header). If not, the save is refused BEFORE the user is asked,
         with a message telling the agent how to fix it.
      3. If the goal did ask for a file, the human must approve it first:
         via terminal input() in the CLI (pre_approved=None, the default), or
         via the Approve/Reject buttons in the web app (pre_approved=True/False).
         If the human declines, the whole run stops immediately.
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
        # Protection 1: only allowed if the goal asked for a file
        if not goal_requires_file_output(state.goal):
            record = StepRecord(
                step_number=state.step_count,
                tool_name=tool_name,
                tool_args=tool_args,
                result=(
                    "BLOCKED: the goal did not ask for a file to be saved, so write_file is not "
                    "allowed. Do not try to save a file; answer from the research already done, "
                    "or search for whatever is still missing."
                ),
                success=False,
            )
            state.history.append(record)
            state.failed_tool_calls += 1
            return state

        # Protection 2: a malformed table is refused before the user is ever asked to approve it
        problem = check_file_content(tool_args.get("path"), tool_args.get("content", ""))
        if problem:
            record = StepRecord(
                step_number=state.step_count,
                tool_name=tool_name,
                tool_args=tool_args,
                result=f"ERROR: not saved - {problem}",
                success=False,
            )
            state.history.append(record)
            state.failed_tool_calls += 1
            return state

        # Protection 3: human approval
        if pre_approved is None:
            # CLI path: ask via terminal
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
            # A deliberate "no" from the user ends the run - the agent must not try again.
            state.done = True
            state.final_answer = DECLINED_MESSAGE
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