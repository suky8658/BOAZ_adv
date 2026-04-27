from langchain_core.messages import AIMessage

from run_agent import _extract_last_message


def test_extract_last_message_handles_ai_message():
    result = {"messages": [AIMessage(content="마트 생성이 필요합니다.")]}

    assert _extract_last_message(result) == "마트 생성이 필요합니다."


def test_extract_last_message_handles_dict_message():
    result = {"messages": [{"role": "assistant", "content": "조회 결과입니다."}]}

    assert _extract_last_message(result) == "조회 결과입니다."
