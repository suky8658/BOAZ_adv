from __future__ import annotations

import argparse
import json

from SQL_Agent.sql_agent.deep_agents.src.sql_agent.sql_agent import build_app

DEFAULT_SCENARIOS = {
    "query": "고객이 가장 많은 상위 5개 주(state)와 각 고객 수를 보여줘.",
    "datamart": (
        "월별 주문 매출과 주문 수를 국가(state) 기준으로 분석할 수 있는 "
        "datamart 테이블을 만들어줘. 테이블명은 mart_monthly_state_sales 로 해줘."
    ),
}


def extract_last_message(result: dict) -> str:
    messages = result.get("messages", [])
    if not messages:
        return "응답이 없습니다."

    last_message = messages[-1]
    content = getattr(last_message, "content", last_message.get("content", ""))
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            else:
                text_parts.append(str(item))
        return "\n".join(part for part in text_parts if part)
    return str(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a smoke scenario against the deepagents SQL agent.")
    parser.add_argument(
        "--scenario",
        choices=sorted(DEFAULT_SCENARIOS),
        default="query",
        help="Predefined scenario to run.",
    )
    parser.add_argument(
        "--question",
        help="Override the predefined question with a custom one.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the full raw agent result.",
    )
    args = parser.parse_args()

    question = args.question or DEFAULT_SCENARIOS[args.scenario]
    app = build_app()
    result = app.invoke({"messages": [{"role": "user", "content": question}]})

    print("[Scenario]")
    print(args.scenario)
    print("\n[Question]")
    print(question)
    print("\n[Last Message]")
    print(extract_last_message(result))

    if args.raw:
        print("\n[Raw Result]")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
