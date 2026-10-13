import asyncio
import json

from langchain_core.messages import HumanMessage, AIMessage

from travel_agent.agent_context import trim_message_window
from travel_agent.agent_context import (
    TravelMemoryMiddleware,
    build_memory_context,
)
from travel_agent.context_summary import (
    build_structured_summary,
    generate_conversation_summary,
)
from travel_agent.repository import SQLiteRepository
from travel_agent.service import ConsultationService


def test_structured_summary_is_deterministic_and_serializable():
    summary = build_structured_summary(
        [HumanMessage(content="去杭州"), AIMessage(content="已记录")],
        {"destination": "杭州", "budget": 8000},
    )
    assert summary == {
        "requirements": {"destination": "杭州", "budget": 8000},
        "message_count": 2,
        "last_message": "已记录",
    }
    json.dumps(summary, ensure_ascii=False)


def test_model_summary_uses_messages_and_returns_text():
    class FakeSummaryModel:
        async def ainvoke(self, messages):
            assert "去杭州" in messages[1].content
            return AIMessage(content="用户计划杭州旅行，预算已确认。")

    result = asyncio.run(
        generate_conversation_summary(
            FakeSummaryModel(), [HumanMessage(content="去杭州")], previous_summary=""
        )
    )
    assert result == "用户计划杭州旅行，预算已确认。"


def test_model_summary_failure_is_safe():
    class BrokenModel:
        async def ainvoke(self, messages):
            raise RuntimeError("offline")

    assert asyncio.run(
        generate_conversation_summary(BrokenModel(), [HumanMessage(content="去杭州")])
    ) is None


def test_summary_input_can_be_the_same_window_boundary_as_context_window():
    messages = [HumanMessage(content=f"消息 {index}") for index in range(20)]
    assert len(trim_message_window(messages, max_messages=12, max_tokens=4000)) == 12


def test_memory_context_unifies_preferences_requirements_and_summaries():
    memory = build_memory_context({
        "customer_preferences": {"budget": 5000, "hotel.star_level": "舒适型"},
        "current_trip_requirements": {"budget": 8000},
        "structured_summary": {"requirements": {"destination": "杭州"}},
        "conversation_summary": "用户计划去杭州。",
    })
    assert memory.effective_preferences == {"hotel.star_level": "舒适型"}
    assert memory.current_requirements == {"budget": 8000}
    assert memory.structured_summary["requirements"] == {"destination": "杭州"}
    assert memory.conversation_summary == "用户计划去杭州。"


def test_memory_middleware_summarizes_only_newly_evicted_messages():
    class CountingSummaryModel:
        def __init__(self):
            self.calls = []

        async def ainvoke(self, messages):
            self.calls.append(messages)
            return AIMessage(content=f"摘要-{len(self.calls)}")

    model = CountingSummaryModel()
    middleware = TravelMemoryMiddleware(model, max_messages=2, max_tokens=4000)
    state = {
        "messages": [HumanMessage(content=f"消息 {index}") for index in range(5)],
        "trip_requirements": {"destination": "杭州"},
    }

    first = asyncio.run(middleware.abefore_model(state, None))
    assert first["summarized_message_count"] == 3
    assert first["structured_summary"]["message_count"] == 3
    assert first["conversation_summary"] == "摘要-1"

    state.update(first)
    assert asyncio.run(middleware.abefore_model(state, None)) is None
    assert len(model.calls) == 1

    state["messages"].append(HumanMessage(content="新消息"))
    second = asyncio.run(middleware.abefore_model(state, None))
    assert second["summarized_message_count"] == 4
    assert len(model.calls) == 2


def test_memory_middleware_falls_back_to_structured_summary():
    class BrokenSummaryModel:
        async def ainvoke(self, messages):
            raise RuntimeError("summary unavailable")

    middleware = TravelMemoryMiddleware(
        BrokenSummaryModel(), max_messages=1, max_tokens=4000
    )
    update = asyncio.run(middleware.abefore_model({
        "messages": [HumanMessage(content="旧消息"), HumanMessage(content="当前消息")],
        "trip_requirements": {"destination": "杭州"},
        "conversation_summary": "已有摘要",
    }, None))
    assert update["structured_summary"]["requirements"] == {"destination": "杭州"}
    assert update["conversation_summary"] == "已有摘要"


def test_chat_persists_both_summaries_when_window_overflows(monkeypatch, tmp_path):
    from tests.test_stage4_preference_context import ContextRecordingModel
    monkeypatch.setattr(
        "travel_agent.agent._import_chat_openai",
        lambda: lambda **kwargs: ContextRecordingModel(),
    )
    monkeypatch.setenv("TRAVEL_AGENT_MODEL_MODE", "openai_compatible")
    monkeypatch.setenv("TRAVEL_AGENT_MODEL_NAME", "offline-context-test")
    for provider in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{provider}_PROVIDER", "sqlite")
    monkeypatch.setenv("TRAVEL_AGENT_CONTEXT_MAX_MESSAGES", "1")
    async def fake_summary(model, messages, *, previous_summary=""):
        return "用户正在规划杭州旅行。"
    monkeypatch.setattr("travel_agent.agent_context.generate_conversation_summary", fake_summary)

    repository = SQLiteRepository(tmp_path / "travel.db")
    service = ConsultationService(repository, tmp_path / "checkpoints.sqlite")
    consultation = service.create_consultation(
        "consultant", "客户", "去杭州", customer_id="customer-1"
    )
    service.chat_consultation(consultation.id, "预算 8000 元")

    async def read_state():
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        async with AsyncSqliteSaver.from_conn_string(str(service.checkpoint_path)) as saver:
            saved = await saver.aget_tuple({"configurable": {"thread_id": consultation.thread_id}})
            return saved.checkpoint["channel_values"]

    state = asyncio.run(read_state())
    assert state["structured_summary"]["requirements"] == {
        "destination": "杭州", "budget": 8000.0
    }
    assert state["conversation_summary"] == "用户正在规划杭州旅行。"
    assert len(repository.get_consultation(consultation.id).messages) == 3
