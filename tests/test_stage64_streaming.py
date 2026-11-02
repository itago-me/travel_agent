"""阶段 6.4：流式输出。

流式是常驻 loop 解锁的第一个能力：`agent.astream` 需要 loop 在请求之间
存活。测试要证明三件事——事件真的分块到达、最终结果与非流式路径一致、
失败时客户端能知道原因。
"""

import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

import travel_agent.agent as agent_module
import travel_agent.service as service_module
from travel_agent.application import create_consultation_service
from travel_agent.service import _stream_events


class _StreamingModel(BaseChatModel):
    """Deterministic model: answers once, no tool calls."""

    reply: str = "已收到需求，正在为您规划行程。"

    @property
    def _llm_type(self) -> str:
        return "stage64-streaming-offline"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = AIMessage(
            content=self.reply,
            usage_metadata={
                "input_tokens": 120,
                "output_tokens": 30,
                "total_tokens": 150,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture
def streaming_service(tmp_path, monkeypatch):
    saver = InMemorySaver()

    class CheckpointContext:
        async def __aenter__(self):
            return saver

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr(
        service_module.AsyncSqliteSaver,
        "from_conn_string",
        lambda _path: CheckpointContext(),
    )
    monkeypatch.setenv("TRAVEL_AGENT_MODEL_MODE", "openai_compatible")
    for provider in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{provider}_PROVIDER", "sqlite")
    model = _StreamingModel()
    monkeypatch.setattr(agent_module, "_import_chat_openai", lambda: (lambda **kw: model))

    service = create_consultation_service(tmp_path / "stream.db")
    return service, model


# --- 事件翻译层 -------------------------------------------------------------


class _Chunk:
    def __init__(self, content=None, tool_call_chunks=None):
        self.content = content
        self.tool_call_chunks = tool_call_chunks or []


def test_text_chunk_becomes_token_event():
    assert _stream_events(_Chunk("你好")) == [{"type": "token", "content": "你好"}]


def test_empty_chunk_produces_no_event():
    """空内容不该产生空事件——否则客户端收到一堆无意义的帧。"""
    assert _stream_events(_Chunk("")) == []
    assert _stream_events(_Chunk(None)) == []


def test_tool_call_chunk_becomes_tool_event():
    chunk = _Chunk(tool_call_chunks=[{"name": "search_hotel_options", "args": "{}"}])

    assert _stream_events(chunk) == [{"type": "tool", "name": "search_hotel_options"}]


def test_structured_content_parts_are_flattened():
    """部分模型把文本切成分段结构。"""
    chunk = _Chunk([{"type": "text", "text": "第一段"}, {"type": "text", "text": "第二段"}])

    assert _stream_events(chunk) == [
        {"type": "token", "content": "第一段"},
        {"type": "token", "content": "第二段"},
    ]


# --- 端到端流式 -------------------------------------------------------------


def test_stream_yields_tokens_then_done(streaming_service):
    service, _model = streaming_service
    consultation = service.create_consultation("c1", "张三", "去杭州")

    events = list(
        service.chat_consultation_stream(consultation.id, "帮我规划", model_name="offline")
    )

    types = [event["type"] for event in events]
    assert "token" in types, f"没有收到任何 token 事件: {types}"
    assert types[-1] == "done"


def test_stream_reassembles_to_the_model_answer(streaming_service):
    """拼起来的 token 必须等于模型的完整回复，一个字不多不少。"""
    service, model = streaming_service
    consultation = service.create_consultation("c1", "张三", "去杭州")

    events = list(
        service.chat_consultation_stream(consultation.id, "帮我规划", model_name="offline")
    )
    text = "".join(e["content"] for e in events if e["type"] == "token")

    assert text == model.reply


def test_stream_result_matches_the_non_stream_shape(streaming_service):
    """流式只是传输方式不同，业务结果必须与非流式逐字段一致。"""
    service, _model = streaming_service
    streamed = service.create_consultation("c1", "张三", "去杭州")
    plain = service.create_consultation("c2", "李四", "去杭州")

    from_stream = list(
        service.chat_consultation_stream(streamed.id, "帮我规划", model_name="offline")
    )[-1]["result"]
    from_plain = service.chat_consultation(plain.id, "帮我规划", model_name="offline")

    # 两条路径走的是同一套 finalize，键集合必须一致（id 相关字段除外）
    assert set(from_stream) == set(from_plain), (
        f"流式多出 {set(from_stream) - set(from_plain)}，"
        f"少了 {set(from_plain) - set(from_stream)}"
    )
    assert from_stream["status"] == from_plain["status"] == "COMPLETED"
    assert from_stream["answer"] == from_plain["answer"]
    assert from_stream["usage"]["total_tokens"] == from_plain["usage"]["total_tokens"]


def test_stream_reports_token_usage(streaming_service):
    """流式不能把成本统计丢掉——否则省了等待却丢了账单。"""
    service, _model = streaming_service
    consultation = service.create_consultation("c1", "张三", "去杭州")

    events = list(
        service.chat_consultation_stream(consultation.id, "帮我规划", model_name="offline")
    )

    usage = events[-1]["result"]["usage"]
    assert usage["reported"] is True
    assert usage["input_tokens"] == 120
    assert usage["output_tokens"] == 30


def test_stream_persists_the_turn(streaming_service):
    """流式跑完必须落库：消息、状态、审计一样都不能少。"""
    service, _model = streaming_service
    consultation = service.create_consultation("c1", "张三", "去杭州")

    list(
        service.chat_consultation_stream(consultation.id, "帮我规划", model_name="offline")
    )

    stored = service.get_consultation(consultation.id)
    assert any(message.role == "user" for message in stored.messages)
    assert any(message.role == "assistant" for message in stored.messages)
    assert service.repository.consult_usage_totals(consultation.id)["turns"] == 1


def test_second_stream_turn_continues_the_thread(streaming_service):
    """第二轮走的是"历史完整"路径，同样必须能流式。"""
    service, _model = streaming_service
    consultation = service.create_consultation("c1", "张三", "去杭州")

    list(
        service.chat_consultation_stream(consultation.id, "第一轮", model_name="offline")
    )
    second = list(
        service.chat_consultation_stream(consultation.id, "第二轮", model_name="offline")
    )

    assert second[-1]["type"] == "done"
    assert second[-1]["result"]["status"] == "COMPLETED"


# --- SSE 传输层 -------------------------------------------------------------


def test_sse_endpoint_emits_well_formed_frames(streaming_service, monkeypatch):
    from fastapi.testclient import TestClient

    from travel_agent.api import create_app

    service, _model = streaming_service
    consultation = service.create_consultation("c1", "张三", "去杭州")
    # 必须在 create_app 之前替换：工厂函数在构建时就实例化了 service，
    # 打补丁太晚会让 API 连到另一个空库，端点于是返回 404。
    monkeypatch.setattr(
        "travel_agent.api.create_consultation_service", lambda _path: service
    )
    client = TestClient(create_app("ignored.db"))

    response = client.post(
        f"/consultations/{consultation.id}/chat/stream",
        json={"message": "帮我规划", "model_name": "offline"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = [
        json.loads(line[len("data: "):])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert frames, "SSE 没有任何数据帧"
    assert frames[-1]["type"] == "done"
