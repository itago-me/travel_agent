import asyncio
from datetime import date, timedelta
from typing import Any
from uuid import uuid4
from pathlib import Path

from .domain import AuditEvent, Consultation, Message
from .mock_model import MockAgentModel
from .requirement_extractor import extract_requirements
from .requirements import TripRequirements
from .repository import SQLiteRepository
from .tool_gateway import (
    ToolAuditEvent,
    ToolExecutionContext,
    ToolPermission,
    create_mock_tool_gateway,
    create_planning_tool_gateway,
)
from .calculators import (
    BudgetCalculationRequest,
    ConflictDetectionRequest,
    ItineraryScoreRequest,
)
from .providers import (
    AttractionOption,
    HotelOption,
    TransportOption,
)
from .planning import MockPlanningModel, run_planning_graph
from .planning_revision import parse_planning_revision


class ConsultationService:
    def __init__(
        self,
        repository: SQLiteRepository,
        checkpoint_path: str | Path | None = None,
    ):
        self.repository = repository
        self.checkpoint_path = Path(
            checkpoint_path
            or repository.database_path.with_name(
                f"{repository.database_path.stem}_checkpoints.sqlite"
            )
        )

    def create_consultation(
        self, consultant_id: str, customer_name: str, initial_message: str
    ) -> Consultation:
        consultation = Consultation(
            id=str(uuid4()),
            thread_id=str(uuid4()),
            consultant_id=consultant_id,
            customer_name=customer_name,
            status="OPEN",
            messages=[Message("user", initial_message)],
        )
        self.repository.create_consultation(consultation)
        self.repository.add_audit_event(
            AuditEvent(
                consultation_id=consultation.id,
                event_type="CONSULTATION_CREATED",
                details={"consultant_id": consultant_id, "thread_id": consultation.thread_id},
            )
        )
        return consultation

    def get_consultation(self, consultation_id: str) -> Consultation:
        return self.repository.get_consultation(consultation_id)

    def append_message(self, consultation_id: str, role: str, content: str) -> None:
        self.repository.append_message(consultation_id, Message(role, content))
        self.repository.add_audit_event(
            AuditEvent(
                consultation_id=consultation_id,
                event_type="MESSAGE_APPENDED",
                details={"role": role},
            )
        )

    def update_requirements(
        self, consultation_id: str, requirements: dict[str, Any]
    ) -> None:
        self.repository.update_requirements(consultation_id, requirements)
        self.repository.add_audit_event(
            AuditEvent(
                consultation_id=consultation_id,
                event_type="REQUIREMENTS_UPDATED",
                details={"fields": sorted(requirements)},
            )
        )

    def resume_consultation(self, consultation_id: str, user_message: str) -> dict[str, Any]:
        consultation = self.get_consultation(consultation_id)
        self.append_message(consultation_id, "user", user_message)

        extracted = extract_requirements(user_message)
        requirements = {**consultation.trip_requirements, **extracted}
        if extracted:
            self.update_requirements(consultation_id, requirements)

        from .workflow import run_requirements_graph

        result = run_requirements_graph(
            MockAgentModel(),
            TripRequirements(**requirements).as_dict(),
            thread_id=consultation.thread_id,
            checkpoint_path=self.checkpoint_path,
        )
        decision = result["decision"]
        self.append_message(consultation_id, "assistant", decision["message"])
        self.repository.add_audit_event(
            AuditEvent(
                consultation_id=consultation_id,
                event_type="AGENT_DECISION",
                details={
                    **decision,
                    "model_call_count": result["model_call_count"],
                    "thread_id": consultation.thread_id,
                },
            )
        )
        return {
            "consultation_id": consultation_id,
            "thread_id": self.get_consultation(consultation_id).thread_id,
            "requirements": requirements,
            "decision": decision,
            "model_call_count": result["model_call_count"],
        }

    def search_options(self, consultation_id: str) -> dict[str, Any]:
        consultation = self.get_consultation(consultation_id)
        requirements = TripRequirements(**consultation.trip_requirements)
        missing_fields = requirements.missing_fields()
        if missing_fields:
            raise ValueError(
                f"旅行需求不完整，缺少字段: {', '.join(missing_fields)}"
            )

        def persist_tool_audit(event: ToolAuditEvent) -> None:
            self.repository.add_audit_event(
                AuditEvent(
                    consultation_id=event.consultation_id,
                    event_type="TOOL_CALLED",
                    details={
                        "thread_id": event.thread_id,
                        "tool_name": event.tool_name,
                        "permission": event.permission.value,
                        "call_number": event.call_number,
                        "outcome": event.outcome,
                        "error_type": event.error_type,
                    },
                )
            )

        gateway = create_mock_tool_gateway(audit_sink=persist_tool_audit)
        context = ToolExecutionContext(
            consultation_id=consultation.id,
            thread_id=consultation.thread_id,
            allowed_permissions={ToolPermission.READ},
        )

        async def execute_searches() -> dict[str, Any]:
            transport = await gateway.execute(
                "search_transport_options",
                {
                    "origin": requirements.origin,
                    "destination": requirements.destination,
                    "departure_date": requirements.start_date,
                    "traveler_count": requirements.traveler_count,
                },
                context,
            )
            hotels = await gateway.execute(
                "search_hotel_options",
                {
                    "city": requirements.destination,
                    "check_in": requirements.start_date,
                    "check_out": requirements.end_date,
                    "rooms": 1,
                },
                context,
            )
            attractions = await gateway.execute(
                "search_attractions",
                {
                    "city": requirements.destination,
                    "visit_date": (
                        date.fromisoformat(requirements.start_date) + timedelta(days=1)
                    ).isoformat(),
                },
                context,
            )
            return {
                "consultation_id": consultation.id,
                "thread_id": consultation.thread_id,
                "transport": transport.output,
                "hotels": hotels.output,
                "attractions": attractions.output,
            }

        return asyncio.run(execute_searches())

    def calculate_options(self, consultation_id: str) -> dict[str, Any]:
        consultation = self.get_consultation(consultation_id)
        requirements = TripRequirements(**consultation.trip_requirements)
        if requirements.missing_fields():
            raise ValueError("旅行需求不完整，无法执行方案计算")
        searched = self.search_options(consultation_id)

        def persist_tool_audit(event: ToolAuditEvent) -> None:
            self.repository.add_audit_event(
                AuditEvent(
                    consultation_id=event.consultation_id,
                    event_type="TOOL_CALLED",
                    details={
                        "thread_id": event.thread_id,
                        "tool_name": event.tool_name,
                        "permission": event.permission.value,
                        "call_number": event.call_number,
                        "outcome": event.outcome,
                        "error_type": event.error_type,
                    },
                )
            )

        transport = TransportOption.model_validate(searched["transport"]["options"][0])
        hotel = HotelOption.model_validate(searched["hotels"]["options"][0])
        attractions = [
            AttractionOption.model_validate(option)
            for option in searched["attractions"]["options"]
        ]
        gateway = create_planning_tool_gateway(audit_sink=persist_tool_audit)
        context = ToolExecutionContext(
            consultation_id=consultation.id,
            thread_id=consultation.thread_id,
            allowed_permissions={ToolPermission.READ, ToolPermission.COMPUTE},
        )

        async def execute_calculations() -> dict[str, Any]:
            budget = await gateway.execute(
                "calculate_trip_budget",
                BudgetCalculationRequest(
                    transport=transport,
                    hotel=hotel,
                    attractions=attractions,
                    check_in=requirements.start_date,
                    check_out=requirements.end_date,
                    budget_limit=requirements.budget,
                ).model_dump(mode="json"),
                context,
            )
            conflicts = await gateway.execute(
                "detect_itinerary_conflicts",
                ConflictDetectionRequest(
                    transport=transport, attractions=attractions
                ).model_dump(mode="json"),
                context,
            )
            score = await gateway.execute(
                "score_trip_options",
                ItineraryScoreRequest(
                    transport=transport,
                    hotel=hotel,
                    attractions=attractions,
                    total_budget=budget.output["total"],
                    budget_limit=requirements.budget,
                    conflict_count=len(conflicts.output["conflicts"]),
                ).model_dump(mode="json"),
                context,
            )
            return {
                "consultation_id": consultation.id,
                "thread_id": consultation.thread_id,
                "budget": budget.output,
                "conflicts": conflicts.output,
                "score": score.output,
            }

        return asyncio.run(execute_calculations())

    def plan_consultation(
        self,
        consultation_id: str,
        *,
        max_tool_calls: int = 6,
    ) -> dict[str, Any]:
        consultation = self.get_consultation(consultation_id)

        def persist_tool_audit(event: ToolAuditEvent) -> None:
            self.repository.add_audit_event(
                AuditEvent(
                    consultation_id=event.consultation_id,
                    event_type="TOOL_CALLED",
                    details={
                        "thread_id": event.thread_id,
                        "tool_name": event.tool_name,
                        "permission": event.permission.value,
                        "call_number": event.call_number,
                        "outcome": event.outcome,
                        "error_type": event.error_type,
                    },
                )
            )

        context = ToolExecutionContext(
            consultation_id=consultation.id,
            thread_id=consultation.thread_id,
            allowed_permissions={ToolPermission.READ, ToolPermission.COMPUTE},
        )
        result = asyncio.run(
            run_planning_graph(
                MockPlanningModel(),
                create_planning_tool_gateway(audit_sink=persist_tool_audit),
                context,
                consultation.trip_requirements,
                consultation.planning_constraints,
                max_tool_calls=max_tool_calls,
            )
        )
        if result["status"] == "COMPLETED" and result.get("itinerary") is not None:
            version_id, version_number = self.repository.save_itinerary_version(
                consultation.id,
                result["itinerary"],
                result["tool_history"],
                result["observations"],
            )
            result["itinerary_version_id"] = version_id
            result["itinerary_version"] = version_number
        event_type = (
            "PLANNING_COMPLETED"
            if result["status"] == "COMPLETED"
            else "PLANNING_STOPPED"
        )
        self.repository.add_audit_event(
            AuditEvent(
                consultation_id=consultation.id,
                event_type=event_type,
                details={
                    "thread_id": consultation.thread_id,
                    "status": result["status"],
                    "tool_call_count": result["tool_call_count"],
                    "model_call_count": result["model_call_count"],
                    "tool_history": result["tool_history"],
                    "last_error": result.get("last_error"),
                },
            )
        )
        return dict(result)

    def revise_consultation(self, consultation_id: str, message: str) -> dict[str, Any]:
        consultation = self.get_consultation(consultation_id)
        self.append_message(consultation_id, "user", message)
        revision = parse_planning_revision(message)
        requirements = dict(consultation.trip_requirements)
        if "budget" in revision:
            requirements["budget"] = revision["budget"]
            self.update_requirements(consultation_id, requirements)
        constraints = {**consultation.planning_constraints, **{
            key: value for key, value in revision.items() if key != "budget"
        }}
        if revision:
            self.repository.update_planning_constraints(consultation_id, constraints)
            self.repository.add_audit_event(
                AuditEvent(
                    consultation_id=consultation_id,
                    event_type="PLANNING_CONSTRAINTS_UPDATED",
                    details={"constraints": constraints},
                )
            )
        result = self.plan_consultation(consultation_id)
        result["revision"] = revision
        return result

    def compare_itineraries(self, consultation_id: str) -> list[dict[str, Any]]:
        versions = self.repository.list_itinerary_versions(consultation_id)
        return [
            {
                "version_id": version["id"],
                "version_number": version["version_number"],
                "budget_total": version["payload"]["budget"]["total"],
                "budget_limit": version["payload"]["budget"]["budget_limit"],
                "score": version["payload"]["score"]["score"],
                "transport_option_id": version["payload"]["transport_option_id"],
                "hotel_option_id": version["payload"]["hotel_option_id"],
                "conflict_count": len(version["payload"]["conflicts"]["conflicts"]),
                "created_at": version["created_at"],
            }
            for version in versions
        ]

    def select_itinerary(self, consultation_id: str, version_id: str) -> dict[str, Any]:
        version = self.repository.get_itinerary_version(consultation_id, version_id)
        self.repository.select_itinerary_version(consultation_id, version_id)
        self.repository.add_audit_event(
            AuditEvent(
                consultation_id=consultation_id,
                event_type="ITINERARY_SELECTED",
                details={"version_id": version_id, "version_number": version["version_number"]},
            )
        )
        return {"version_id": version_id, "version_number": version["version_number"], "payload": version["payload"]}
