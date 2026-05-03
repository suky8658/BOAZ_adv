"""
Subagent 기반 아키텍처 디버그
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from SQL_Agent.sql_agent.deep_agents.src.sql_agent.sql_agent import build_agent
from dotenv import load_dotenv

load_dotenv()

agent = build_agent()

question = "주문 건수가 가장 많은 고객 상위 5명은 누구야?"
print(f"\n[질문] {question}")

result = agent.invoke({
    "messages": [{"role": "user", "content": question}]
})

print("\n[메시지 흐름 요약]")
for i, msg in enumerate(result.get("messages", [])):
    mtype = type(msg).__name__
    tc = [t.get("name") for t in getattr(msg, "tool_calls", [])]
    content_preview = str(msg.content)[:300]
    add_kw = getattr(msg, "additional_kwargs", {})
    if tc:
        print(f"  [{i}] {mtype} → tool_calls: {tc}")
    elif "function_call" in add_kw:
        print(f"  [{i}] {mtype} → function_call (add_kwargs): {add_kw.get('function_call', {}).get('name')}")
    elif content_preview.strip():
        print(f"  [{i}] {mtype}: {content_preview}")
    else:
        print(f"  [{i}] {mtype}: (empty)")

print(f"\n[최종 답변]\n{result['messages'][-1].content}")
