from dataclasses import dataclass, field
from datetime import datetime, timezone
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


@dataclass
class Consultation:
    id: str
    thread_id: str
    consultant_id: str
    customer_name: str
    status: str
    messages: list[Message] = field(default_factory=list)
    trip_requirements: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
