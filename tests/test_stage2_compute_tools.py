import asyncio
from datetime import date

from travel_agent.calculators import BudgetCalculationRequest
from travel_agent.mock_providers import MockHotelProvider, MockTransportProvider
from travel_agent.providers import HotelSearchRequest, TransportSearchRequest
from travel_agent.tool_gateway import (
    ToolExecutionContext,
    ToolPermission,
    create_planning_tool_gateway,
)


def test_planning_gateway_exposes_compute_tools_with_compute_permission():
    gateway = create_planning_tool_gateway()

    compute_tools = [tool for tool in gateway.list_tools() if tool.permission == ToolPermission.COMPUTE]

    assert [tool.name for tool in compute_tools] == [
        "calculate_trip_budget",
        "detect_itinerary_conflicts",
        "score_trip_options",
    ]


def test_gateway_executes_budget_calculation_only_with_compute_permission():
    async def run():
        transport = await MockTransportProvider().search(
            TransportSearchRequest(origin="北京", destination="杭州", departure_date=date(2026, 10, 1), traveler_count=2)
        )
        hotel = await MockHotelProvider().search(
            HotelSearchRequest(city="杭州", check_in=date(2026, 10, 1), check_out=date(2026, 10, 4), rooms=1)
        )
        gateway = create_planning_tool_gateway()
        return await gateway.execute(
            "calculate_trip_budget",
            BudgetCalculationRequest(
                transport=transport.options[0], hotel=hotel.options[0], attractions=[],
                check_in=date(2026, 10, 1), check_out=date(2026, 10, 4), budget_limit=8000,
            ).model_dump(mode="json"),
            ToolExecutionContext("c1", "t1", {ToolPermission.COMPUTE}),
        )

    result = asyncio.run(run())
    assert result.output["within_budget"] is True
