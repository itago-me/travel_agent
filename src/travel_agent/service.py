from typing import Any
from uuid import uuid4
from pathlib import Path

from .domain import AuditEvent, Consultation, Message
from .mock_model import MockAgentModel
from .requirement_extractor import extract_requirements
from .requirements import TripRequirements
from .repository import SQLiteRepository


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
