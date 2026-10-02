import asyncio

import pytest
from pydantic import ValidationError

from travel_agent.tool_gateway import (
    PermissionDeniedError,
    ToolExecutionContext,
    ToolPermission,
    UnknownToolError,
    create_mock_tool_gateway,
)


def test_gateway_lists_registered_read_tools():
    gateway = create_mock_tool_gateway()

    tools = gateway.list_tools()

    assert [tool.name for tool in tools] == [
        "search_attractions",
        "search_hotel_options",
        "search_transport_options",
    ]
    assert {tool.permission for tool in tools} == {ToolPermission.READ}


def test_gateway_validates_arguments_and_executes_provider():
    events = []
    gateway = create_mock_tool_gateway(audit_sink=events.append)
    context = ToolExecutionContext(
        consultation_id="consultation-1",
        thread_id="thread-1",
        allowed_permissions={ToolPermission.READ},
    )

    result = asyncio.run(
        gateway.execute(
            "search_transport_options",
            {
                "origin": "北京",
                "destination": "杭州",
                "departure_date": "2026-10-01",
                "traveler_count": 2,
                "mode": "train",
            },
            context,
        )
    )

    assert result.tool_name == "search_transport_options"
    assert result.call_number == 1
    assert len(result.output["options"]) == 2
    assert events[-1].outcome == "SUCCEEDED"


def test_gateway_rejects_invalid_arguments_before_provider_execution():
    gateway = create_mock_tool_gateway()
    context = ToolExecutionContext(
        consultation_id="consultation-1",
        thread_id="thread-1",
        allowed_permissions={ToolPermission.READ},
    )

    with pytest.raises(ValidationError):
        asyncio.run(
            gateway.execute(
                "search_transport_options",
                {
                    "origin": "北京",
                    "destination": "杭州",
                    "departure_date": "2026-10-01",
                    "traveler_count": 0,
                },
                context,
            )
        )


def test_gateway_rejects_unknown_or_disallowed_tools():
    gateway = create_mock_tool_gateway()
    denied = ToolExecutionContext(
        consultation_id="consultation-1",
        thread_id="thread-1",
        allowed_permissions=set(),
    )

    with pytest.raises(UnknownToolError):
        asyncio.run(gateway.execute("missing_tool", {}, denied))
    with pytest.raises(PermissionDeniedError):
        asyncio.run(
            gateway.execute(
                "search_attractions",
                {"city": "杭州", "visit_date": "2026-10-02"},
                denied,
            )
        )
