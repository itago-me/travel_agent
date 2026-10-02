from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .calculators import (
    BudgetCalculationRequest,
    ConflictDetectionRequest,
    ItineraryScoreRequest,
)
from .providers import AttractionOption, HotelOption, TransportOption
from .requirements import TripRequirements
from .tool_gateway import ToolExecutionContext, ToolGateway


TOOL_SEQUENCE = (
    "search_transport_options",
    "search_hotel_options",
    "search_attractions",
    "calculate_trip_budget",
    "detect_itinerary_conflicts",
    "score_trip_options",
)


class IncompletePlanningRequirementsError(ValueError):
    pass


class PlanningState(TypedDict, total=False):
    requirements: dict[str, str | int | float]
    constraints: dict[str, str | float]
    observations: dict[str, dict[str, Any]]
    next_tool: str | None
    status: str
    tool_call_count: int
    model_call_count: int
    max_tool_calls: int
    tool_history: list[str]
    itinerary: dict[str, Any] | None
    last_error: str | None


@dataclass(frozen=True)
class PlanningDecision:
    action: Literal["CALL_TOOL", "COMPLETE", "STOP"]
    tool_name: str | None = None
    status: str = "RUNNING"


class MockPlanningModel:
    """Deterministically chooses the next tool from current observations."""

    def decide(self, state: PlanningState) -> PlanningDecision:
        if state.get("last_error"):
            return PlanningDecision(action="STOP", status="FAILED")
        observations = state.get("observations", {})
        for tool_name in TOOL_SEQUENCE:
            if tool_name not in observations:
                if state.get("tool_call_count", 0) >= state["max_tool_calls"]:
                    return PlanningDecision(action="STOP", status="STOPPED_LIMIT")
                return PlanningDecision(action="CALL_TOOL", tool_name=tool_name)
        return PlanningDecision(action="COMPLETE", status="COMPLETED")


def build_planning_graph(
    model: MockPlanningModel,
    gateway: ToolGateway,
    context: ToolExecutionContext,
):
    async def decide_next_action(state: PlanningState) -> dict[str, Any]:
        decision = model.decide(state)
        update: dict[str, Any] = {
            "model_call_count": state.get("model_call_count", 0) + 1,
            "next_tool": decision.tool_name,
            "status": decision.status,
        }
        if decision.action == "COMPLETE":
            update["itinerary"] = _build_itinerary(
                state["observations"], state.get("constraints", {})
            )
        return update

    async def execute_tool(state: PlanningState) -> dict[str, Any]:
        tool_name = state["next_tool"]
        if tool_name not in TOOL_SEQUENCE:
            return {
                "last_error": f"Tool is not allowed in planning workflow: {tool_name}",
                "next_tool": None,
            }
        try:
            result = await gateway.execute(
                tool_name,
                _build_tool_arguments(
                    tool_name,
                    state["requirements"],
                    state.get("constraints", {}),
                    state.get("observations", {}),
                ),
                context,
            )
        except Exception as error:
            return {
                "last_error": f"{type(error).__name__}: {error}",
                "next_tool": None,
            }
        observations = dict(state.get("observations", {}))
        observations[tool_name] = result.output
        return {
            "observations": observations,
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "tool_history": [*state.get("tool_history", []), tool_name],
            "next_tool": None,
        }

    async def route_after_decision(state: PlanningState) -> str:
        return "execute_tool" if state.get("next_tool") else END

    graph = StateGraph(PlanningState)
    graph.add_node("decide_next_action", decide_next_action)
    graph.add_node("execute_tool", execute_tool)
    graph.add_edge(START, "decide_next_action")
    graph.add_conditional_edges("decide_next_action", route_after_decision)
    graph.add_edge("execute_tool", "decide_next_action")
    return graph.compile()


async def run_planning_graph(
    model: MockPlanningModel,
    gateway: ToolGateway,
    context: ToolExecutionContext,
    requirements: dict[str, str | int | float],
    constraints: dict[str, str | float] | None = None,
    *,
    max_tool_calls: int = 6,
) -> PlanningState:
    normalized = TripRequirements(**requirements)
    missing = normalized.missing_fields()
    if missing:
        raise IncompletePlanningRequirementsError(
            f"旅行需求不完整，缺少字段: {', '.join(missing)}"
        )
    graph = build_planning_graph(model, gateway, context)
    return await graph.ainvoke(
        {
            "requirements": normalized.as_dict(),
            "constraints": constraints or {},
            "observations": {},
            "next_tool": None,
            "status": "RUNNING",
            "tool_call_count": 0,
            "model_call_count": 0,
            "max_tool_calls": max_tool_calls,
            "tool_history": [],
            "itinerary": None,
            "last_error": None,
        }
    )


def _build_tool_arguments(
    tool_name: str,
    requirements: dict[str, str | int | float],
    constraints: dict[str, str | float],
    observations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if tool_name == "search_transport_options":
        return {
            "origin": requirements["origin"],
            "destination": requirements["destination"],
            "departure_date": requirements["start_date"],
            "traveler_count": requirements["traveler_count"],
            **({"mode": constraints["transport_mode"]} if constraints.get("transport_mode") else {}),
        }
    if tool_name == "search_hotel_options":
        return {
            "city": requirements["destination"],
            "check_in": requirements["start_date"],
            "check_out": requirements["end_date"],
            "rooms": 1,
        }
    if tool_name == "search_attractions":
        visit_date = date.fromisoformat(str(requirements["start_date"])) + timedelta(
            days=1
        )
        return {
            "city": requirements["destination"],
            "visit_date": visit_date.isoformat(),
        }

    transport, hotel, attractions = _selected_options(observations, constraints)
    if tool_name == "calculate_trip_budget":
        return BudgetCalculationRequest(
            transport=transport,
            hotel=hotel,
            attractions=attractions,
            check_in=requirements["start_date"],
            check_out=requirements["end_date"],
            budget_limit=requirements["budget"],
        ).model_dump(mode="json")
    if tool_name == "detect_itinerary_conflicts":
        return ConflictDetectionRequest(
            transport=transport,
            attractions=attractions,
        ).model_dump(mode="json")
    if tool_name == "score_trip_options":
        budget = observations["calculate_trip_budget"]
        conflicts = observations["detect_itinerary_conflicts"]
        return ItineraryScoreRequest(
            transport=transport,
            hotel=hotel,
            attractions=attractions,
            total_budget=budget["total"],
            budget_limit=requirements["budget"],
            conflict_count=len(conflicts["conflicts"]),
        ).model_dump(mode="json")
    raise ValueError(f"Unsupported planning tool: {tool_name}")


def _selected_options(
    observations: dict[str, dict[str, Any]],
    constraints: dict[str, str | float] | None = None,
) -> tuple[TransportOption, HotelOption, list[AttractionOption]]:
    transport_options = observations["search_transport_options"]["options"]
    hotel_options = observations["search_hotel_options"]["options"]
    attraction_options = observations["search_attractions"]["options"]
    if not transport_options or not hotel_options:
        raise ValueError("供应商没有返回可用于规划的交通或酒店选项")
    hotels = [HotelOption.model_validate(option) for option in hotel_options]
    preference = (constraints or {}).get("hotel_preference")
    if preference == "highest_rating":
        hotel = max(hotels, key=lambda option: option.rating)
    elif preference == "lowest_price":
        hotel = min(hotels, key=lambda option: option.price_per_night)
    else:
        hotel = hotels[0]
    return (
        TransportOption.model_validate(transport_options[0]),
        hotel,
        [AttractionOption.model_validate(option) for option in attraction_options],
    )


def _build_itinerary(observations: dict[str, dict[str, Any]], constraints=None) -> dict[str, Any]:
    transport, hotel, attractions = _selected_options(observations, constraints)
    return {
        "transport_option_id": transport.option_id,
        "hotel_option_id": hotel.option_id,
        "attraction_option_ids": [item.option_id for item in attractions],
        "budget": observations["calculate_trip_budget"],
        "conflicts": observations["detect_itinerary_conflicts"],
        "score": observations["score_trip_options"],
    }
