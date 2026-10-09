import json

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from travel_agent import cli


class OfflineCheckpointer:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def aget_tuple(self, config):
        return None


def test_cli_chat_passes_factory_tools_to_agent_and_reports_tool_call(
    monkeypatch, tmp_path, capsys
):
    captured = {}
    monkeypatch.setattr(
        "travel_agent.service.AsyncSqliteSaver.from_conn_string",
        lambda _path: OfflineCheckpointer(),
    )

    class FakeChatModel:
        def __init__(self, **kwargs):
            captured["model_kwargs"] = kwargs

    class FakeCompiledAgent:
        def __init__(self, tools):
            self.tools = {tool.name: tool for tool in tools}

        async def ainvoke(self, payload, *, config):
            captured["input"] = payload
            captured["config"] = config
            observation = await self.tools["search_hotel_options"].ainvoke(
                {
                    "city": "杭州",
                    "check_in": "2026-10-15",
                    "check_out": "2026-10-17",
                    "rooms": 1,
                }
            )
            assert observation["provider"] == "sqlite_hotel"
            return {
                "messages": [
                    AIMessage(
                        content="已查询酒店",
                        tool_calls=[
                            {
                                "name": "search_hotel_options",
                                "args": {
                                    "city": "杭州",
                                    "check_in": "2026-10-15",
                                    "check_out": "2026-10-17",
                                    "rooms": 1,
                                },
                                "id": "call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="杭州酒店查询完成"),
                ]
            }

    monkeypatch.setenv("TRAVEL_AGENT_MODEL_MODE", "openai_compatible")
    monkeypatch.setenv("TRAVEL_AGENT_MODEL_NAME", "offline-demo")
    monkeypatch.setenv("TRAVEL_AGENT_HOTEL_PROVIDER", "sqlite")
    monkeypatch.setenv("TRAVEL_AGENT_TRANSPORT_PROVIDER", "sqlite")
    monkeypatch.setenv("TRAVEL_AGENT_ATTRACTION_PROVIDER", "sqlite")
    monkeypatch.setattr("travel_agent.agent._import_chat_openai", lambda: FakeChatModel)

    def fake_create_agent(**kwargs):
        captured["agent_kwargs"] = kwargs
        return FakeCompiledAgent(kwargs["tools"])

    monkeypatch.setattr("travel_agent.agent._import_create_agent", lambda: fake_create_agent)

    database = tmp_path / "travel.db"
    checkpoints = tmp_path / "checkpoints.sqlite"
    assert (
        cli.main(
            [
                "--database",
                str(database),
                "--checkpoint-database",
                str(checkpoints),
                "start",
                "--consultant-id",
                "consultant-1",
                "--customer-name",
                "张先生",
                "--message",
                "我想去杭州旅行",
            ]
        )
        == 0
    )
    consultation_id = json.loads(capsys.readouterr().out)["consultation_id"]

    assert (
        cli.main(
            [
                "--database",
                str(database),
                "--checkpoint-database",
                str(checkpoints),
                "chat",
                consultation_id,
                "--message",
                "请帮我查杭州酒店",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)

    assert result["answer"] == "杭州酒店查询完成"
    assert result["tool_calls"] == ["search_hotel_options"]
    assert captured["config"]["configurable"]["thread_id"] == result["thread_id"]
    assert {tool.name for tool in captured["agent_kwargs"]["tools"]} == {
        "search_transport_options",
        "search_hotel_options",
        "search_attractions",
    }


def test_service_runs_official_agent_with_async_travel_tool(monkeypatch, tmp_path):
    from travel_agent.application import create_consultation_service

    class ToolCallingModel(BaseChatModel):
        @property
        def _llm_type(self):
            return "offline-tool-calling"

        def bind_tools(self, tools, **kwargs):
            return self

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            return self._generate(messages, stop=stop, **kwargs)

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            observations = [m for m in messages if isinstance(m, ToolMessage)]
            if observations:
                result = json.loads(observations[-1].content)
                assert result["provider"] == "sqlite_hotel"
                message = AIMessage(content="已读取酒店工具结果")
            else:
                message = AIMessage(content="", tool_calls=[{
                    "name": "search_hotel_options",
                    "args": {"city": "杭州", "check_in": "2026-10-15",
                             "check_out": "2026-10-17", "rooms": 1},
                    "id": "hotel-call",
                    "type": "tool_call",
                }])
            return ChatResult(generations=[ChatGeneration(message=message)])

    class CheckpointContext:
        async def __aenter__(self):
            return InMemorySaver()

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr("travel_agent.agent._import_chat_openai", lambda: lambda **kw: ToolCallingModel())
    monkeypatch.setattr("travel_agent.service.AsyncSqliteSaver.from_conn_string", lambda path: CheckpointContext())
    monkeypatch.setenv("TRAVEL_AGENT_MODEL_MODE", "openai_compatible")
    for name in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{name}_PROVIDER", "sqlite")
    service = create_consultation_service(tmp_path / "travel.db")
    consultation = service.create_consultation("consultant-1", "测试客户", "去杭州")
    result = service.chat_consultation(consultation.id, "查酒店", model_name="offline")
    assert result["answer"] == "已读取酒店工具结果"
    assert result["tool_calls"] == ["search_hotel_options"]
    assert any(m["type"] == "tool" for m in result["messages"])
