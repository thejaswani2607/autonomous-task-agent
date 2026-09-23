from langgraph.graph import StateGraph, END
from state import AgentState
from nodes.planner import plan_next_action
from nodes.executor import execute_action
from nodes.evaluator import evaluate_progress


def planner_node(state: AgentState) -> AgentState:
    """Wraps plan_next_action: asks Gemini for the next move, stores it for the Executor."""
    state.llm_call_count += 1
    try:
        decision = plan_next_action(state)
        state.pending_action = decision
    except Exception as e:
        state.aborted = True
        state.done = True
        state.final_answer = (
            f"Run stopped: Gemini API was unavailable even after retries ({e}). "
            f"This is usually a temporary free-tier capacity issue — try running again in a few minutes."
        )
    return state


def executor_node(state: AgentState) -> AgentState:
    """Wraps execute_action: runs whatever the Planner just decided."""
    decision = state.pending_action or {}
    tool_name = decision.get("tool_name")
    tool_args = decision.get("tool_args", {})
    state = execute_action(state, tool_name, tool_args)
    state.pending_action = None
    return state


def evaluator_node(state: AgentState) -> AgentState:
    """Wraps evaluate_progress: judges the last result, may set done=True."""
    state.llm_call_count += 1
    try:
        state = evaluate_progress(state)
    except Exception as e:
        state.done = True
        state.final_answer = (
            f"Run stopped: Gemini API was unavailable even after retries during evaluation ({e}). "
            f"The last completed step is saved in history — try running again in a few minutes."
        )
        if state.history:
            state.history[-1].evaluator_verdict = "stuck"
            state.history[-1].evaluator_reasoning = "Evaluator could not run due to a persistent Gemini API outage."
    return state


def route_after_planner(state: AgentState) -> str:
    """If the Planner couldn't get a decision at all, skip straight to the end."""
    if state.aborted:
        return END
    return "executor"


def route_after_evaluator(state: AgentState) -> str:
    """The main loop decision: loop back, or stop?"""
    if state.done:
        return END
    return "planner"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("evaluator", evaluator_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"executor": "executor", END: END},
    )
    graph.add_edge("executor", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {END: END, "planner": "planner"},
    )

    return graph.compile()