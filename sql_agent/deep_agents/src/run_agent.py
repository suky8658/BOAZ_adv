from __future__ import annotations

from typing import Any
from uuid import uuid4

from langgraph.types import Command, StateSnapshot

from SQL_Agent.sql_agent.deep_agents.src.sql_agent.sql_agent import build_app

app = build_app()

INTERESTING_NODES = {
    "model_request": "모델 추론",
    "tools": "도구 실행",
}


def _extract_last_message(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    if not messages:
        return "응답이 없습니다."

    last_message = messages[-1]
    if isinstance(last_message, dict):
        content = last_message.get("content", "")
    else:
        content = getattr(last_message, "content", "")

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            else:
                text_parts.append(str(item))
        return "\n".join(part for part in text_parts if part)
    return str(content)


def _source_label(ns: tuple[str, ...]) -> str:
    if not ns:
        return "main"
    if any(segment.startswith("tools:") for segment in ns):
        return "subagent"
    return "agent"


def _format_tool_args(args: dict[str, Any]) -> str:
    parts = []
    for key, value in args.items():
        rendered = str(value)
        if len(rendered) > 180:
            rendered = rendered[:177] + "..."
        parts.append(f"{key}={rendered}")
    return ", ".join(parts)


def _print_step_header(title: str) -> None:
    print(f"\n[{title}]")


def _print_stream_chunk(chunk: dict[str, Any], seen_events: set[tuple[Any, ...]]) -> None:
    source = _source_label(chunk["ns"])

    if chunk["type"] == "updates":
        for node_name, data in chunk["data"].items():
            if node_name not in INTERESTING_NODES:
                continue

            event_key = ("updates", chunk["ns"], node_name)
            if event_key in seen_events:
                continue
            seen_events.add(event_key)

            print(f"  - {source:8} | {INTERESTING_NODES[node_name]}")

            if not chunk["ns"] and node_name == "model_request":
                for msg in data.get("messages", []):
                    for tool_call in getattr(msg, "tool_calls", []):
                        if tool_call["name"] == "task":
                            subagent_type = tool_call["args"].get("subagent_type", "subagent")
                            description = tool_call["args"].get("description", "")
                            print(f"    -> 위임: {subagent_type}")
                            if description:
                                print(f"       {description}")

    elif chunk["type"] == "messages":
        token, _metadata = chunk["data"]

        for tool_call_chunk in getattr(token, "tool_call_chunks", []) or []:
            tool_name = tool_call_chunk.get("name")
            if not tool_name:
                continue

            event_key = ("tool_call", chunk["ns"], tool_name, tool_call_chunk.get("id"))
            if event_key in seen_events:
                continue
            seen_events.add(event_key)
            print(f"  - {source:8} | 도구 호출 제안: {tool_name}")

        if getattr(token, "type", None) == "tool":
            tool_name = getattr(token, "name", "tool")
            event_key = ("tool_result", chunk["ns"], tool_name, getattr(token, "tool_call_id", None))
            if event_key in seen_events:
                return
            seen_events.add(event_key)
            print(f"  - {source:8} | 도구 완료: {tool_name}")


def _stream_until_pause_or_finish(input_value: Any, config: dict[str, Any]) -> StateSnapshot:
    seen_events: set[tuple[Any, ...]] = set()
    _print_step_header("Progress")

    for chunk in app.stream(
        input_value,
        config=config,
        stream_mode=["updates", "messages"],
        subgraphs=True,
        version="v2",
    ):
        _print_stream_chunk(chunk, seen_events)

    return app.get_state(config)


def _prompt_interrupt_decisions(snapshot: StateSnapshot) -> Command:
    interrupt_value = snapshot.interrupts[0].value
    action_requests = interrupt_value["action_requests"]

    _print_step_header("Approval Required")
    decisions = []

    for index, action in enumerate(action_requests, start=1):
        print(f"{index}. tool: {action['name']}")
        print(f"   args: {_format_tool_args(action['args'])}")

        while True:
            answer = input("   승인할까요? [y/N]: ").strip().lower()
            if answer in {"y", "yes"}:
                decisions.append({"type": "approve"})
                break
            if answer in {"", "n", "no"}:
                decisions.append({"type": "reject"})
                break

    return Command(resume={"decisions": decisions})


def _run_question(user_question: str) -> str:
    config = {"configurable": {"thread_id": str(uuid4())}}
    pending_input: Any = {"messages": [{"role": "user", "content": user_question}]}

    while True:
        snapshot = _stream_until_pause_or_finish(pending_input, config)
        if snapshot.interrupts:
            pending_input = _prompt_interrupt_decisions(snapshot)
            continue
        values = snapshot.values if isinstance(snapshot.values, dict) else {}
        return _extract_last_message(values)


if __name__ == "__main__":
    while True:
        user_question = input("\n질문 입력 (종료하려면 exit): ").strip()
        if user_question.lower() == "exit":
            break

        print("\n" + "=" * 60)
        print("[Question]")
        print(user_question)

        final_message = _run_question(user_question)

        _print_step_header("Final Answer")
        print(final_message)
        print("=" * 60)
