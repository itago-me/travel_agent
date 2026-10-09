from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from types import SimpleNamespace

from travel_agent.domain import Message
from travel_agent.repository import SQLiteRepository
from travel_agent.service import ConsultationService
from travel_agent.service import _tool_history_status


class FakeAgent:
    def __init__(self, calls, checkpointer):
        self.calls = calls
        self.checkpointer = checkpointer

    async def ainvoke(self, payload, *, config):
        self.calls.append((payload, config))
        self.checkpointer.threads.add(config["configurable"]["thread_id"])
        return {"messages": [AIMessage(content=f"已处理第 {len(self.calls)} 轮") ]}

    async def aget_state(self, config):
        return SimpleNamespace(values={"messages": []}, next=())


class FakeCheckpointer:
    def __init__(self):
        self.threads = set()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    async def aget_tuple(self, config):
        if config["configurable"]["thread_id"] in self.threads:
            return object()
        return None


class FakeCheckpointerContext:
    def __init__(self, checkpointer):
        self.checkpointer = checkpointer

    async def __aenter__(self):
        return self.checkpointer

    async def __aexit__(self, *_args):
        return False


def test_chat_uses_same_thread_and_only_submits_new_message(monkeypatch, tmp_path):
    repository = SQLiteRepository(tmp_path / "travel.db")
    service = ConsultationService(repository, checkpoint_path=tmp_path / "checkpoints.sqlite")
    consultation = service.create_consultation("consultant-1", "张先生", "我想去杭州玩 4 天")
    calls = []
    checkpointer = FakeCheckpointer()
    fake_agent = FakeAgent(calls, checkpointer)

    monkeypatch.setattr("travel_agent.service.create_agent_app", lambda *args, **kwargs: fake_agent)
    monkeypatch.setattr(
        "travel_agent.service.AsyncSqliteSaver.from_conn_string",
        lambda _path: FakeCheckpointerContext(checkpointer),
    )
    monkeypatch.setattr(
        "travel_agent.service.load_agent_settings",
        lambda *_args: type("Settings", (), {"model_mode": "openai_compatible", "model_name": "demo", "api_key": None, "base_url": None})(),
    )

    first = service.chat_consultation(consultation.id, "请先帮我确认需要哪些信息")
    second = service.chat_consultation(consultation.id, "从北京出发，预算 8000 元")

    assert first["answer"] == "已处理第 1 轮"
    assert second["answer"] == "已处理第 2 轮"
    assert [call[0]["messages"] for call in calls] == [
        [
            {"role": "user", "content": "我想去杭州玩 4 天"},
            {"role": "user", "content": "请先帮我确认需要哪些信息"},
        ],
        [{"role": "user", "content": "从北京出发，预算 8000 元"}],
    ]
    assert calls[0][1]["configurable"]["thread_id"] == consultation.thread_id
    assert calls[1][1]["configurable"]["thread_id"] == consultation.thread_id

    restored = service.get_consultation(consultation.id)
    assert [message.content for message in restored.messages] == [
        "我想去杭州玩 4 天",
        "请先帮我确认需要哪些信息",
        "已处理第 1 轮",
        "从北京出发，预算 8000 元",
        "已处理第 2 轮",
    ]


def test_chat_builds_travel_tools_and_explicitly_injects_them_into_agent(
    monkeypatch, tmp_path
):
    repository = SQLiteRepository(tmp_path / "travel.db")
    service = ConsultationService(
        repository, checkpoint_path=tmp_path / "checkpoints.sqlite"
    )
    consultation = service.create_consultation(
        "consultant-1", "张先生", "我想去杭州旅行"
    )
    checkpointer = FakeCheckpointer()
    fake_agent = FakeAgent([], checkpointer)
    tools = [object(), object(), object()]
    captured = {}

    monkeypatch.setenv("TRAVEL_AGENT_HOTEL_PROVIDER", "flyai")
    monkeypatch.setenv("TRAVEL_AGENT_TRANSPORT_PROVIDER", "flyai")
    monkeypatch.setenv("TRAVEL_AGENT_ATTRACTION_PROVIDER", "flyai")
    monkeypatch.setattr(
        "travel_agent.service.load_agent_settings",
        lambda *_args: type(
            "Settings",
            (),
            {
                "model_mode": "openai_compatible",
                "model_name": "demo",
                "api_key": None,
                "base_url": None,
            },
        )(),
    )
    monkeypatch.setattr(
        "travel_agent.service.AsyncSqliteSaver.from_conn_string",
        lambda _path: FakeCheckpointerContext(checkpointer),
    )

    def fake_create_travel_tools(database, provider_settings):
        captured["tool_database"] = database
        captured["provider_settings"] = provider_settings
        return tools

    def fake_create_agent_app(database, settings, **kwargs):
        captured["agent_database"] = database
        captured["agent_tools"] = kwargs.get("tools")
        return fake_agent

    monkeypatch.setattr(
        "travel_agent.service.create_travel_tools", fake_create_travel_tools
    )
    monkeypatch.setattr(
        "travel_agent.service.create_agent_app", fake_create_agent_app
    )

    service.chat_consultation(consultation.id, "请查询酒店")

    assert captured["tool_database"] == repository.database_path
    assert captured["provider_settings"].hotel == "flyai"
    assert captured["provider_settings"].transport == "flyai"
    assert captured["provider_settings"].attraction == "flyai"
    assert captured["agent_database"] == repository.database_path
    assert captured["agent_tools"] is tools


def test_tool_history_status_detects_pending_and_invalid_sequences():
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "search_hotel_options", "args": {}, "id": "call-1"}],
    )

    assert _tool_history_status([tool_call]) == "pending"
    assert (
        _tool_history_status(
            [tool_call, ToolMessage(content="{}", tool_call_id="call-1")]
        )
        == "complete"
    )
    assert _tool_history_status([tool_call, HumanMessage(content="重试")]) == "invalid"


def test_chat_repairs_invalid_tool_history_from_business_messages(
    monkeypatch, tmp_path
):
    repository = SQLiteRepository(tmp_path / "travel.db")
    service = ConsultationService(repository, checkpoint_path=tmp_path / "cp.sqlite")
    consultation = service.create_consultation("c1", "客户", "去杭州")
    checkpointer = FakeCheckpointer()
    checkpointer.threads.add(consultation.thread_id)
    calls = []
    updates = []

    class RepairingAgent(FakeAgent):
        async def aget_state(self, config):
            return SimpleNamespace(
                values={
                    "messages": [
                        AIMessage(content="", tool_calls=[{
                            "name": "search_hotel_options", "args": {}, "id": "call-1"
                        }]),
                        HumanMessage(content="再次查询"),
                    ]
                },
                next=("model",),
            )

        async def aupdate_state(self, config, values, *, as_node):
            updates.append((values, as_node))

    agent = RepairingAgent(calls, checkpointer)
    monkeypatch.setattr("travel_agent.service.create_agent_app", lambda *a, **k: agent)
    monkeypatch.setattr(
        "travel_agent.service.AsyncSqliteSaver.from_conn_string",
        lambda _path: FakeCheckpointerContext(checkpointer),
    )
    monkeypatch.setattr(
        "travel_agent.service.load_agent_settings",
        lambda *_: SimpleNamespace(
            model_mode="openai_compatible", model_name="demo", api_key=None, base_url=None
        ),
    )

    result = service.chat_consultation(consultation.id, "再次查询")

    assert result["answer"] == "已处理第 1 轮"
    assert calls[0][0] is None
    assert updates[0][1] == "__start__"
    reset_messages = updates[0][0]["messages"]
    assert isinstance(reset_messages[0], RemoveMessage)
    assert reset_messages[-1] == {"role": "user", "content": "再次查询"}


def test_chat_resumes_pending_tool_before_submitting_new_user_message(
    monkeypatch, tmp_path
):
    repository = SQLiteRepository(tmp_path / "travel.db")
    service = ConsultationService(repository, checkpoint_path=tmp_path / "cp.sqlite")
    consultation = service.create_consultation("c1", "客户", "去杭州")
    checkpointer = FakeCheckpointer()
    checkpointer.threads.add(consultation.thread_id)
    calls = []

    class PendingAgent(FakeAgent):
        async def aget_state(self, config):
            return SimpleNamespace(
                values={
                    "messages": [
                        AIMessage(content="", tool_calls=[{
                            "name": "search_hotel_options", "args": {}, "id": "call-1"
                        }])
                    ]
                },
                next=("tools",),
            )

    agent = PendingAgent(calls, checkpointer)
    monkeypatch.setattr("travel_agent.service.create_agent_app", lambda *a, **k: agent)
    monkeypatch.setattr(
        "travel_agent.service.AsyncSqliteSaver.from_conn_string",
        lambda _path: FakeCheckpointerContext(checkpointer),
    )
    monkeypatch.setattr(
        "travel_agent.service.load_agent_settings",
        lambda *_: SimpleNamespace(
            model_mode="openai_compatible", model_name="demo", api_key=None, base_url=None
        ),
    )

    service.chat_consultation(consultation.id, "继续查询")

    assert calls[0][0] is None
    assert calls[1][0] == {
        "messages": [{"role": "user", "content": "继续查询"}]
    }
