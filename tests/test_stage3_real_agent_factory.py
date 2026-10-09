from pathlib import Path

from travel_agent.agent import AgentSettings, create_agent_app, load_agent_settings
from travel_agent.sqlite_providers import initialize_travel_data


def test_agent_settings_read_openai_compatible_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAVEL_AGENT_MODEL_MODE", "openai_compatible")
    monkeypatch.setenv("TRAVEL_AGENT_MODEL_NAME", "demo-model")
    monkeypatch.setenv("TRAVEL_AGENT_API_KEY", "secret")
    monkeypatch.setenv("TRAVEL_AGENT_BASE_URL", "https://example.test/v1")

    settings = load_agent_settings(tmp_path / ".env")

    assert settings == AgentSettings(
        model_mode="openai_compatible",
        model_name="demo-model",
        api_key="secret",
        base_url="https://example.test/v1",
    )


def test_create_agent_uses_official_factory_and_stage31_tools(monkeypatch, tmp_path):
    database = tmp_path / "travel.db"
    initialize_travel_data(database)
    calls = {}

    class FakeChatModel:
        def __init__(self, **kwargs):
            calls["model_kwargs"] = kwargs

    def fake_import_chat_openai():
        return FakeChatModel

    def fake_create_agent(**kwargs):
        calls["agent_kwargs"] = kwargs
        return "compiled-agent"

    monkeypatch.setattr("travel_agent.agent._import_chat_openai", fake_import_chat_openai)
    monkeypatch.setattr("travel_agent.agent._import_create_agent", lambda: fake_create_agent)

    result = create_agent_app(
        database,
        AgentSettings(
            model_mode="openai_compatible",
            model_name="demo-model",
            api_key="secret",
            base_url="https://example.test/v1",
        ),
    )

    assert result == "compiled-agent"
    assert calls["model_kwargs"] == {
        "model": "demo-model",
        "api_key": "secret",
        "base_url": "https://example.test/v1",
    }
    assert {tool.name for tool in calls["agent_kwargs"]["tools"]} == {
        "search_transport_options",
        "search_hotel_options",
        "search_attractions",
    }


def test_create_agent_explicitly_passes_provider_settings_to_tools(monkeypatch, tmp_path):
    captured = {}

    class FakeChatModel:
        def __init__(self, **kwargs):
            pass

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setenv("TRAVEL_AGENT_HOTEL_PROVIDER", "flyai")
    monkeypatch.setenv("TRAVEL_AGENT_TRANSPORT_PROVIDER", "sqlite")
    monkeypatch.setenv("TRAVEL_AGENT_ATTRACTION_PROVIDER", "sqlite")
    monkeypatch.setattr("travel_agent.agent._import_chat_openai", lambda: FakeChatModel)
    monkeypatch.setattr("travel_agent.agent._import_create_agent", lambda: fake_create_agent)
    monkeypatch.setattr(
        "travel_agent.langchain_tools.create_travel_tools",
        lambda database, provider_settings=None, **kwargs: captured.update(
            {"provider_settings": provider_settings}
        )
        or [],
    )

    create_agent_app(tmp_path / "travel.db", AgentSettings(model_name="demo-model"))

    assert captured["provider_settings"].hotel == "flyai"
    assert captured["provider_settings"].transport == "sqlite"
    assert captured["provider_settings"].attraction == "sqlite"


def test_create_agent_passes_langgraph_checkpointer(monkeypatch, tmp_path):
    database = tmp_path / "travel.db"
    initialize_travel_data(database)
    checkpointer = object()
    calls = {}

    class FakeChatModel:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr("travel_agent.agent._import_chat_openai", lambda: FakeChatModel)
    monkeypatch.setattr(
        "travel_agent.agent._import_create_agent",
        lambda: lambda **kwargs: calls.update(kwargs) or "agent",
    )

    create_agent_app(
        database,
        AgentSettings(model_name="demo-model"),
        checkpointer=checkpointer,
    )

    assert calls["checkpointer"] is checkpointer
