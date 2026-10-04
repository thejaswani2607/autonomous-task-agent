import os
import re
import streamlit as st

from state import AgentState, StepRecord
from graph import build_graph  # not used directly, but confirms the core loop still imports cleanly
from nodes.planner import plan_next_action
from nodes.executor import execute_action
from nodes.evaluator import evaluate_progress
from memory import search_memory, add_memory
from main import save_run_log, _format_memory_context
from checks import goal_requires_file_output, is_genuine_success
from tools import check_file_content

st.set_page_config(page_title="Autonomous Task Agent", page_icon="🤖", layout="centered")

MAX_STEPS = 15

st.markdown("""
<style>
.approval-box {
    background-color: #EEF2FF;
    border: 1px solid #C7D2FE;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 14px;
}
.processing-box {
    background-color: #EEF2FF;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-weight: 500;
    color: #1E3A8A;
}
</style>
""", unsafe_allow_html=True)


def init_session():
    if "state" not in st.session_state:
        st.session_state.state = None
    if "awaiting_approval" not in st.session_state:
        st.session_state.awaiting_approval = False
    if "pending_decision" not in st.session_state:
        st.session_state.pending_decision = None
    if "memory_note" not in st.session_state:
        st.session_state.memory_note = None


def reset_session():
    st.session_state.state = None
    st.session_state.awaiting_approval = False
    st.session_state.pending_decision = None
    st.session_state.memory_note = None


def _safe_plan(state: AgentState):
    """Ask the Planner. If the Gemini API is down even after retries, end the run
    cleanly with an explanation instead of crashing the page. Returns None in that case."""
    state.llm_call_count += 1
    try:
        return plan_next_action(state)
    except Exception as e:
        state.aborted = True
        state.done = True
        state.final_answer = (
            f"Run stopped: Gemini API was unavailable even after retries ({e}). "
            f"This is usually a temporary free-tier capacity issue - try again in a few minutes."
        )
        return None


def _safe_evaluate(state: AgentState) -> AgentState:
    """Ask the Evaluator, with the same clean-failure behavior as _safe_plan."""
    state.llm_call_count += 1
    try:
        return evaluate_progress(state)
    except Exception as e:
        state.done = True
        state.final_answer = (
            f"Run stopped: Gemini API was unavailable even after retries during evaluation ({e}). "
            f"The completed steps are saved - try again in a few minutes."
        )
        if state.history:
            state.history[-1].evaluator_verdict = "stuck"
            state.history[-1].evaluator_reasoning = "Evaluator could not run due to a persistent Gemini API outage."
        return state


def _finalize_if_done(state: AgentState):
    """Whenever a run reaches done, always save its transcript to logs/ (independent of
    the download button). Only GENUINE successes are added to memory - not declined,
    aborted, or step-limit runs."""
    if state.done:
        save_run_log(state)
        if is_genuine_success(state):
            add_memory(state.goal, state.final_answer)


def render_history(state: AgentState):
    for step in state.history:
        with st.container(border=True):
            status = "✅ Success" if step.success else "❌ Failed"
            st.markdown(f"**Step {step.step_number}: `{step.tool_name}`** — {status}")
            st.caption(str(step.tool_args))
            st.text(step.result[:400] + ("..." if len(step.result) > 400 else ""))
            if step.evaluator_verdict:
                st.markdown(f"*Evaluator verdict: **{step.evaluator_verdict}** — {step.evaluator_reasoning}*")


def find_last_output_file(state: AgentState):
    for step in reversed(state.history):
        if step.tool_name == "write_file" and step.success:
            # the path is everything between "file to " and the trailing "(N characters)" / "(N rows)",
            # so file names containing spaces are handled too
            match = re.search(r"Wrote (?:Excel )?file to (.+?) \(\d+ (?:characters|rows)\)", step.result)
            if match:
                return match.group(1)
    return None


st.title("🤖 Autonomous Task Agent")
st.caption("Give it a goal in plain English. It will plan and search, and if your goal asks for a file, "
           "it will pause for your approval before saving anything.")

init_session()

# -------------------- Screen 1: no run in progress --------------------
if st.session_state.state is None:
    goal = st.text_input(
        "What should the agent do?",
        placeholder='e.g. Find 3 cheap wireless earbuds under $100 and save the results as earbuds.csv',
    )
    if st.button("Run Agent", type="primary", disabled=not goal.strip()):
        matches = search_memory(goal)
        memory_context = _format_memory_context(matches)
        if matches:
            st.session_state.memory_note = f"Recalled {len(matches)} similar past run(s) from memory."
        st.session_state.state = AgentState(goal=goal, memory_context=memory_context)
        st.rerun()

# -------------------- Screen 2: a run is in progress or finished --------------------
else:
    state = st.session_state.state

    if st.session_state.memory_note:
        st.info(st.session_state.memory_note)

    st.subheader(f"Goal: {state.goal}")
    render_history(state)

    # --- Finished ---
    if state.done:
        if is_genuine_success(state):
            st.success("Run complete")
        else:
            st.info("Run stopped - the goal was not completed")
        st.markdown(f"**Final answer:** {state.final_answer}")
        st.markdown(
            f"Steps: {state.step_count} | LLM calls: {state.llm_call_count} "
            f"| Tool successes: {state.successful_tool_calls} | Tool failures: {state.failed_tool_calls}"
        )
        st.caption("This run's transcript has already been saved to logs/ - independent of the download button below.")

        output_path = find_last_output_file(state)
        if output_path and os.path.exists(output_path):
            with open(output_path, "rb") as f:
                st.download_button("Download result file", f, file_name=os.path.basename(output_path))

        if st.button("Start New Run"):
            reset_session()
            st.rerun()

    # --- Waiting for the user to approve/reject a file write ---
    elif st.session_state.awaiting_approval:
        decision = st.session_state.pending_decision
        tool_args = decision.get("tool_args", {})

        st.markdown('<div class="approval-box">', unsafe_allow_html=True)
        st.markdown("**The agent wants to write a file. Review before approving.**")
        st.markdown(f"**Path:** `{tool_args.get('path')}`")
        st.code(tool_args.get("content", "")[:1000], language="text")
        st.caption("If you reject this, the run will stop here - it will not try again.")
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        if col1.button("✅ Approve", type="primary"):
            state = execute_action(state, decision["tool_name"], tool_args, pre_approved=True)
            state = _safe_evaluate(state)
            st.session_state.state = state
            st.session_state.awaiting_approval = False
            st.session_state.pending_decision = None
            _finalize_if_done(state)
            st.rerun()

        if col2.button("❌ Reject"):
            # The executor ends the run itself when the user declines
            state = execute_action(state, decision["tool_name"], tool_args, pre_approved=False)
            st.session_state.state = state
            st.session_state.awaiting_approval = False
            st.session_state.pending_decision = None
            _finalize_if_done(state)  # saves the transcript; a declined run is not added to memory
            st.rerun()

    # --- Mid-run: take the next automatic step ---
    else:
        if state.step_count >= MAX_STEPS:
            state.done = True
            state.final_answer = "Stopped: reached the 15-step safety limit before the goal was confirmed complete."
            st.session_state.state = state
            _finalize_if_done(state)
            st.rerun()
        else:
            st.markdown(
                f'<div class="processing-box">⏳ Working on step {state.step_count + 1}... '
                f'this can take 20-60 seconds per step, please keep this tab open.</div>',
                unsafe_allow_html=True,
            )
            with st.spinner("Thinking..."):
                decision = _safe_plan(state)

                if decision is None:
                    # Gemini was unreachable: the run has been ended cleanly
                    st.session_state.state = state
                    _finalize_if_done(state)
                    st.rerun()

                tool_name = decision.get("tool_name")
                tool_args = decision.get("tool_args", {})

                if (
                    tool_name == "write_file"
                    and goal_requires_file_output(state.goal)
                    and not check_file_content(tool_args.get("path"), tool_args.get("content", ""))
                ):
                    # Goal asked for a file and the table is valid: pause and show the Approve / Reject buttons
                    st.session_state.pending_decision = decision
                    st.session_state.awaiting_approval = True
                    st.session_state.state = state
                    st.rerun()
                else:
                    # Normal step. (If the Planner tries write_file although the goal never asked
                    # for a file, or with a malformed table, the executor refuses it and records why -
                    # no approval screen is shown for a save that would be wrong.)
                    state = execute_action(state, tool_name, tool_args)
                    if not state.done:
                        state = _safe_evaluate(state)
                    st.session_state.state = state
                    _finalize_if_done(state)
                    st.rerun()