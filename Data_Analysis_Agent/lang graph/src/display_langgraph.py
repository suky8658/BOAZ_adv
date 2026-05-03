from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict):
    user_question: str
    csv_paths: List[str]
    dataset_context: str
    analysis_plan: Dict[str, Any]
    python_draft: Dict[str, Any]
    execution_result: Dict[str, Any]
    validation: Dict[str, Any]
    retry_count: int
    max_retries: int
    feedback: str
    error: str
    final_answer: str


def load_context(state: AgentState):
    return {}


def plan_analysis(state: AgentState):
    return {}


def generate_python_code(state: AgentState):
    return {}


def execute_analysis(state: AgentState):
    return {}


def validate_result(state: AgentState):
    return {}


def increase_retry(state: AgentState):
    return {}


def finalize_answer(state: AgentState):
    return {}


def route_after_validation(state: AgentState):
    return "finalize"


def build_display_app():
    graph = StateGraph(AgentState)

    graph.add_node("load_context", load_context)
    graph.add_node("plan_analysis", plan_analysis)
    graph.add_node("generate_python_code", generate_python_code)
    graph.add_node("execute_analysis", execute_analysis)
    graph.add_node("validate_result", validate_result)
    graph.add_node("increase_retry", increase_retry)
    graph.add_node("finalize_answer", finalize_answer)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "plan_analysis")
    graph.add_edge("plan_analysis", "generate_python_code")
    graph.add_edge("generate_python_code", "execute_analysis")
    graph.add_edge("execute_analysis", "validate_result")

    graph.add_conditional_edges(
        "validate_result",
        route_after_validation,
        {
            "retry": "increase_retry",
            "finalize": "finalize_answer",
        },
    )

    graph.add_edge("increase_retry", "generate_python_code")
    graph.add_edge("finalize_answer", END)

    return graph.compile()


if __name__ == "__main__":
    from pathlib import Path

    output_dir = Path(__file__).resolve().parents[1] / "graph_output"
    output_dir.mkdir(exist_ok=True)

    app = build_display_app()
    graph = app.get_graph()

    mermaid_path = output_dir / "analysis_agent_langgraph.mmd"
    png_path = output_dir / "analysis_agent_langgraph.png"

    mermaid_path.write_text(graph.draw_mermaid(), encoding="utf-8")
    png_path.write_bytes(graph.draw_mermaid_png())

    print(f"Mermaid saved to: {mermaid_path}")
    print(f"PNG saved to: {png_path}")
