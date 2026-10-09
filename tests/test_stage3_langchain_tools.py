import asyncio
from datetime import date

from travel_agent.langchain_tools import create_travel_tools
from travel_agent.sqlite_providers import initialize_travel_data


def test_travel_tools_expose_agent_selectable_names_and_schemas(tmp_path):
    database = tmp_path / "travel.db"
    initialize_travel_data(database)

    tools = create_travel_tools(database)

    assert [tool.name for tool in tools] == [
        "search_transport_options",
        "search_hotel_options",
        "search_attractions",
    ]
    assert all(tool.args_schema is not None for tool in tools)


def test_travel_tool_returns_structured_observation_for_agent(tmp_path):
    database = tmp_path / "travel.db"
    initialize_travel_data(database)
    tools = {tool.name: tool for tool in create_travel_tools(database)}

    result = asyncio.run(
        tools["search_hotel_options"].ainvoke(
            {
                "city": "杭州",
                "check_in": date(2026, 10, 1).isoformat(),
                "check_out": date(2026, 10, 4).isoformat(),
                "rooms": 1,
            }
        )
    )

    assert result["provider"] == "sqlite_hotel"
    assert {option["name"] for option in result["options"]} == {
        "西湖假日酒店",
        "钱塘精选酒店",
    }
