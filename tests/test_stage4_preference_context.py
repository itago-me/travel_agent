"""Offline checks through the real Agent, service, CLI and SQLite checkpoint."""

import json
import asyncio

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from travel_agent.cli import main
from travel_agent.repository import SQLiteRepository
from travel_agent.service import ConsultationService
from travel_agent.agent_context import resolve_preference_conflicts
from travel_agent.agent_context import trim_message_window


class ContextRecordingModel(BaseChatModel):
    """Expose the actual model input; does not simulate semantic reasoning."""

    @property
    def _llm_type(self):
        return "context-recording"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content=json.dumps([
                {"type": message.type, "content": message.content}
                for message in messages
            ], ensure_ascii=False)
        ))])


def test_explicit_current_requirements_remove_only_conflicting_defaults():
    effective = resolve_preference_conflicts(
        {
            "budget": 5000,
            "hotel.star_level": "舒适型",
            "transport.preferred_mode": "高铁",
        },
        {"budget": 8000},
    )
    assert effective == {
        "hotel.star_level": "舒适型",
        "transport.preferred_mode": "高铁",
    }


def test_message_window_keeps_recent_messages_and_respects_message_limit():
    messages = [HumanMessage(content=f"第 {index} 轮") for index in range(5)]
    kept = trim_message_window(messages, max_messages=2, max_tokens=4000)
    assert [message.content for message in kept] == ["第 3 轮", "第 4 轮"]


def test_message_window_respects_estimated_token_limit():
    messages = [HumanMessage(content="a" * 20), HumanMessage(content="b" * 20)]
    kept = trim_message_window(messages, max_messages=12, max_tokens=9)
    assert [message.content for message in kept] == ["b" * 20]


def test_message_window_does_not_start_with_orphan_tool_result():
    messages = [
        HumanMessage(content="查询酒店"),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "1"}]),
        ToolMessage(content="酒店结果", tool_call_id="1"),
        HumanMessage(content="继续比较"),
    ]
    kept = trim_message_window(messages, max_messages=2, max_tokens=4000)
    assert [message.type for message in kept] == ["ai", "tool", "human"]


def test_message_window_keeps_complete_parallel_tool_result_group():
    messages = [
        HumanMessage(content="查询交通和酒店"),
        AIMessage(content="", tool_calls=[
            {"name": "transport", "args": {}, "id": "transport-1"},
            {"name": "hotel", "args": {}, "id": "hotel-1"},
        ]),
        ToolMessage(content="交通结果", tool_call_id="transport-1"),
        ToolMessage(content="酒店结果", tool_call_id="hotel-1"),
        HumanMessage(content="继续比较"),
    ]
    kept = trim_message_window(messages, max_messages=2, max_tokens=4000)
    assert [message.type for message in kept] == ["ai", "tool", "tool", "human"]
    assert [message.tool_call_id for message in kept[1:3]] == [
        "transport-1", "hotel-1"
    ]


def test_agent_middleware_applies_configured_window_to_model_input(
    monkeypatch, tmp_path
):
    configure_offline_model(monkeypatch)
    monkeypatch.setenv("TRAVEL_AGENT_CONTEXT_MAX_MESSAGES", "2")
    from langgraph.checkpoint.memory import InMemorySaver
    from travel_agent.agent import AgentSettings, create_agent_app

    agent = create_agent_app(
        tmp_path / "travel.db", AgentSettings(model_name="offline"),
        tools=[], checkpointer=InMemorySaver(),
    )
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "旧消息 1"},
            {"role": "user", "content": "旧消息 2"},
            {"role": "user", "content": "当前消息 1"},
            {"role": "user", "content": "当前消息 2"},
        ]
    }, {"configurable": {"thread_id": "window-thread"}})
    model_input = json.loads(result["messages"][-1].content)
    assert [message["content"] for message in model_input if message["type"] == "human"] == [
        "当前消息 1", "当前消息 2"
    ]


def configure_offline_model(monkeypatch):
    monkeypatch.setattr(
        "travel_agent.agent._import_chat_openai",
        lambda: lambda **kwargs: ContextRecordingModel(),
    )
    monkeypatch.setenv("TRAVEL_AGENT_MODEL_MODE", "openai_compatible")
    monkeypatch.setenv("TRAVEL_AGENT_MODEL_NAME", "offline-context-test")
    for provider in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{provider}_PROVIDER", "sqlite")


def test_preferences_reach_model_and_persist_as_customer_scoped_snapshot(
    monkeypatch, tmp_path
):
    configure_offline_model(monkeypatch)
    database = tmp_path / "travel.db"
    checkpoints = tmp_path / "checkpoints.sqlite"
    repository = SQLiteRepository(database)
    repository.set_customer_preference("alice", "hotel.star_level", "舒适型")
    repository.set_customer_preference("alice", "transport.preferred_mode", "高铁")
    repository.set_customer_preference("bob", "hotel.star_level", "豪华型")
    service = ConsultationService(repository, checkpoints)
    consultation = service.create_consultation(
        "consultant", "Alice", "杭州旅行", customer_id="alice"
    )

    first = service.chat_consultation(consultation.id, "请推荐酒店")
    model_input = json.loads(first["answer"])
    system = model_input[0]["content"]
    assert "中国境内旅行顾问" in system  # original prompt survives
    assert "舒适型" in system
    assert "高铁" in system  # no keyword relevance filtering
    assert "豪华型" not in system
    assert "当前" in system and "优先" in system
    assert [m["content"] for m in model_input if m["type"] == "human"] == [
        "杭州旅行", "请推荐酒店"
    ]  # preferences are not fabricated user messages

    repository.set_customer_preference("alice", "hotel.star_level", "经济型")
    # New service and repository mimic a subsequent CLI process.
    restarted = ConsultationService(SQLiteRepository(database), checkpoints)
    second = restarted.chat_consultation(consultation.id, "这次坐飞机")
    assert "舒适型" in json.loads(second["answer"])[0]["content"]
    assert "经济型" not in json.loads(second["answer"])[0]["content"]
    assert "杭州旅行" in second["answer"]

    other = restarted.create_consultation(
        "consultant", "Bob", "杭州旅行", customer_id="bob"
    )
    result = restarted.chat_consultation(other.id, "推荐酒店")
    assert "豪华型" in json.loads(result["answer"])[0]["content"]
    assert "舒适型" not in result["answer"]

    new = restarted.create_consultation(
        "consultant", "Alice", "新的旅行", customer_id="alice"
    )
    result = restarted.chat_consultation(new.id, "推荐酒店")
    assert "经济型" in json.loads(result["answer"])[0]["content"]

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async def read_snapshot():
        async with AsyncSqliteSaver.from_conn_string(str(checkpoints)) as saver:
            saved = await saver.aget_tuple({
                "configurable": {"thread_id": consultation.thread_id}
            })
            return saved.checkpoint["channel_values"]

    state = asyncio.run(read_snapshot())
    assert state["customer_id"] == "alice"
    assert state["customer_preferences"] == {
        "hotel.star_level": "舒适型", "transport.preferred_mode": "高铁"
    }
    assert not any(message.type == "system" for message in state["messages"])


def test_cli_chat_without_preferences_keeps_normal_model_context(
    monkeypatch, tmp_path, capsys
):
    configure_offline_model(monkeypatch)
    common = ["--database", str(tmp_path / "travel.db")]
    assert main(common + [
        "start", "--consultant-id", "c1", "--customer-id", "new-customer",
        "--customer-name", "新客户", "--message", "去杭州"
    ]) == 0
    consultation_id = json.loads(capsys.readouterr().out)["consultation_id"]
    assert main(common + [
        "chat", consultation_id, "--message", "推荐酒店"
    ]) == 0
    response = json.loads(capsys.readouterr().out)
    model_input = json.loads(response["answer"])
    assert "中国境内旅行顾问" in model_input[0]["content"]
    assert [m["content"] for m in model_input if m["type"] == "human"] == [
        "去杭州", "推荐酒店"
    ]


def test_sync_agent_injects_structured_values_without_persisting_system_messages(
    monkeypatch, tmp_path
):
    configure_offline_model(monkeypatch)
    from langgraph.checkpoint.memory import InMemorySaver
    from travel_agent.agent import AgentSettings, create_agent_app

    agent = create_agent_app(
        tmp_path / "travel.db", AgentSettings(model_name="offline"),
        tools=[], checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "sync-thread"}}
    result = agent.invoke({
        "customer_id": "alice",
        "customer_preferences": {"hotel.needs": {"quiet": True}},
        "messages": [{"role": "user", "content": "推荐酒店"}],
    }, config)
    assert '"quiet": true' in json.loads(result["messages"][-1].content)[0]["content"]
    assert not any(m.type == "system" for m in result["messages"])


def test_current_requirement_is_visible_and_conflicting_preference_is_hidden(
    monkeypatch, tmp_path
):
    configure_offline_model(monkeypatch)
    from travel_agent.agent import AgentSettings, create_agent_app
    from langgraph.checkpoint.memory import InMemorySaver

    agent = create_agent_app(
        tmp_path / "travel.db", AgentSettings(model_name="offline"),
        tools=[], checkpointer=InMemorySaver(),
    )
    result = agent.invoke({
        "customer_id": "alice",
        "customer_preferences": {"budget": 5000, "hotel.star_level": "舒适型"},
        "current_trip_requirements": {"budget": 8000},
        "messages": [{"role": "user", "content": "这次预算 8000 元"}],
    }, {"configurable": {"thread_id": "conflict-thread"}})
    system = json.loads(result["messages"][-1].content)[0]["content"]
    assert '"budget": 8000' in system
    assert '"budget": 5000' not in system
    assert "舒适型" in system


def test_service_extracts_current_budget_before_agent_call(monkeypatch, tmp_path):
    configure_offline_model(monkeypatch)
    repository = SQLiteRepository(tmp_path / "travel.db")
    repository.set_customer_preference("alice", "budget", 5000)
    repository.set_customer_preference("alice", "hotel.star_level", "舒适型")
    service = ConsultationService(repository, tmp_path / "checkpoints.sqlite")
    consultation = service.create_consultation(
        "consultant", "Alice", "杭州旅行", customer_id="alice"
    )

    result = service.chat_consultation(consultation.id, "这次预算 8000 元")
    system = json.loads(result["answer"])[0]["content"]
    assert '"budget": 8000.0' in system
    assert '"budget": 5000' not in system
    assert "舒适型" in system


def test_pre44_checkpoint_gets_preferences_on_next_chat(monkeypatch, tmp_path):
    configure_offline_model(monkeypatch)
    from langchain.agents import create_agent
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    repository = SQLiteRepository(tmp_path / "travel.db")
    checkpoints = tmp_path / "checkpoints.sqlite"
    service = ConsultationService(repository, checkpoints)
    consultation = service.create_consultation(
        "c1", "Alice", "杭州旅行", customer_id="alice"
    )
    repository.set_customer_preference("alice", "hotel.star_level", "舒适型")

    async def create_old_checkpoint():
        async with AsyncSqliteSaver.from_conn_string(str(checkpoints)) as saver:
            old_agent = create_agent(ContextRecordingModel(), tools=[], checkpointer=saver)
            await old_agent.ainvoke({
                "messages": [{"role": "user", "content": "杭州旅行"}]
            }, {"configurable": {"thread_id": consultation.thread_id}})

    asyncio.run(create_old_checkpoint())
    result = service.chat_consultation(consultation.id, "推荐酒店")
    model_input = json.loads(result["answer"])
    assert "舒适型" in model_input[0]["content"]
    assert [m["content"] for m in model_input if m["type"] == "human"] == [
        "杭州旅行", "推荐酒店"
    ]
