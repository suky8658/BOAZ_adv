from pathlib import Path


FALLBACK_MERMAID = """---
title: Analysis Agent 0416
---
flowchart TD
    START([START]) --> load_context[load_context]
    load_context --> plan_analysis[plan_analysis]
    plan_analysis --> generate_python_code[generate_python_code]
    generate_python_code --> execute_analysis[execute_analysis]
    execute_analysis --> validate_result[validate_result]
    validate_result -->|retry| increase_retry[increase_retry]
    increase_retry --> generate_python_code
    validate_result -->|finalize| finalize_answer[finalize_answer]
    finalize_answer --> END([END])
"""


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "graph_output"
    output_dir.mkdir(exist_ok=True)

    mermaid_path = output_dir / "analysis_agent_graph.mmd"
    png_path = output_dir / "analysis_agent_graph.png"
    graph = None

    try:
        from analysis_agent import build_app

        app = build_app()
        graph = app.get_graph()
        mermaid = graph.draw_mermaid()
    except Exception as exc:
        mermaid = FALLBACK_MERMAID
        print(f"LangGraph import skipped: {exc}")
        print("Writing the graph structure declared in analysis_agent.py instead.")

    mermaid_path.write_text(mermaid, encoding="utf-8")
    print(f"Mermaid graph saved to: {mermaid_path}")

    try:
        if graph is None:
            raise RuntimeError("LangGraph is not available in this Python environment.")
        png_path.write_bytes(graph.draw_mermaid_png())
        print(f"PNG graph saved to: {png_path}")
    except Exception as exc:
        print(f"PNG export skipped: {exc}")
        print("You can still paste the .mmd file into a Mermaid viewer.")


if __name__ == "__main__":
    main()
