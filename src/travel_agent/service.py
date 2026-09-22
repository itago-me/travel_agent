from typing import Any
from uuid import uuid4

from .domain import AuditEvent, Consultation, Message
from .repository import SQLiteRepository


class ConsultationService:
    def __init__(self, repository: SQLiteRepository):
        self.repository = repository

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
