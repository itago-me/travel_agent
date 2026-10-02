import asyncio

import pytest

from travel_agent.planning import (
    IncompletePlanningRequirementsError,
    MockPlanningModel,
    run_planning_graph,
)
from travel_agent.tool_gateway import (
    ToolExecutionContext,
    ToolPermission,
    create_planning_tool_gateway,
)


COMPLETE_REQUIREMENTS = {
    "origin": "北京",
    "destination": "杭州",
    "start_date": "2026-10-01",
    "end_date": "2026-10-04",
    "traveler_count": 2,
    "budget": 8000,
}


def run_graph(requirements=COMPLETE_REQUIREMENTS, max_tool_calls=6):
    return asyncio.run(
        run_planning_graph(
            MockPlanningModel(),
            create_planning_tool_gateway(),
            ToolExecutionContext(
                consultation_id="consultation-1",
                thread_id="thread-1",
                allowed_permissions={ToolPermission.READ, ToolPermission.COMPUTE},
            ),
            requirements,
            max_tool_calls=max_tool_calls,
        )
    )


def test_planning_graph_calls_tools_in_observation_driven_order():
    result = run_graph()

    assert result["status"] == "COMPLETED"
    assert result["tool_call_count"] == 6
    assert result["tool_history"] == [
        "search_transport_options",
        "search_hotel_options",
        "search_attractions",
        "calculate_trip_budget",
        "detect_itinerary_conflicts",
        "score_trip_options",
    ]
    assert result["itinerary"]["budget"]["total"] == 3915.0
    assert result["itinerary"]["score"]["score"] == 83.7


def test_planning_graph_rejects_incomplete_requirements_before_search():
    with pytest.raises(IncompletePlanningRequirementsError):
        run_graph({"destination": "杭州"})


def test_planning_graph_stops_when_tool_call_limit_is_reached():
    result = run_graph(max_tool_calls=2)

    assert result["status"] == "STOPPED_LIMIT"
    assert result["tool_call_count"] == 2
    assert result["tool_history"] == [
        "search_transport_options",
        "search_hotel_options",
    ]
    assert result["itinerary"] is None
