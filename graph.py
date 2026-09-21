from langgraph.graph import StateGraph, END
from state import AgentState
from nodes.planner import plan_next_action
from nodes.executor import execute_action
from nodes.evaluator import evaluate_progress


def planner_node(state: AgentState) -> AgentState:
    """Wraps plan_next_action: asks Gemini for the next move, stores it for the Executor."""
    state.llm_call_count += 1
    decision = plan_next_action(state)
    state.pending_action = decision
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
    state = evaluate_progress(state)
    return state


def route_after_evaluator(state: AgentState) -> str:
    """The one decision point in the whole graph: loop back, or stop?"""
    if state.done:
        return END
    return "planner"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("evaluator", evaluator_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {END: END, "planner": "planner"},
    )

    return graph.compile()