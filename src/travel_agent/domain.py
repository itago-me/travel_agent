from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class AuditEvent:
    consultation_id: str
    event_type: str
    details: dict[str, Any]
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class CustomerPreference:
    customer_id: str
    preference_key: str
    value: Any
    source: str
    created_at: str
    updated_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "preference_key": self.preference_key,
            "value": self.value,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ApprovalStatus(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class ApprovalDecision(StrEnum):
    APPROVE = "APPROVE"
    EDIT = "EDIT"
    REJECT = "REJECT"


class BookingStatus(StrEnum):
    CREATED = "CREATED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ApprovalRequest:
    id: str
    consultation_id: str
    thread_id: str
    tool_name: str
    tool_arguments: dict[str, Any]
    risk_level: str
    status: ApprovalStatus
    requested_by: str
    created_at: str
    interrupt_id: str | None = None
    action_index: int | None = None
    decision: ApprovalDecision | None = None
    decided_by: str | None = None
    decision_reason: str | None = None
    edited_arguments: dict[str, Any] | None = None
    decided_at: str | None = None
    expires_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "consultation_id": self.consultation_id,
            "thread_id": self.thread_id,
            "interrupt_id": self.interrupt_id,
            "action_index": self.action_index,
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments,
            "risk_level": self.risk_level,
            "status": self.status.value,
            "requested_by": self.requested_by,
            "decision": self.decision.value if self.decision else None,
            "decided_by": self.decided_by,
            "decision_reason": self.decision_reason,
            "edited_arguments": self.edited_arguments,
            "created_at": self.created_at,
            "decided_at": self.decided_at,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class MockBooking:
    id: str
    consultation_id: str
    itinerary_version_id: str
    total_amount: float
    status: BookingStatus
    idempotency_key: str
    created_at: str
    updated_at: str
    cancellation_idempotency_key: str | None = None
    cancellation_reason: str | None = None
    cancelled_at: str | None = None


@dataclass
class Consultation:
    id: str
    thread_id: str
    consultant_id: str
    customer_name: str
    status: str
    customer_id: str = ""
    messages: list[Message] = field(default_factory=list)
    trip_requirements: dict[str, Any] = field(default_factory=dict)
    planning_constraints: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
