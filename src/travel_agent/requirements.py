from dataclasses import dataclass


REQUIRED_FIELDS = (
    "origin",
    "destination",
    "start_date",
    "end_date",
    "traveler_count",
    "budget",
)


@dataclass
class TripRequirements:
    origin: str | None = None
    destination: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    traveler_count: int | None = None
    budget: float | None = None

    def missing_fields(self) -> list[str]:
        return [field for field in REQUIRED_FIELDS if getattr(self, field) in (None, "")]

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            field: value
            for field in REQUIRED_FIELDS
            if (value := getattr(self, field)) not in (None, "")
        }


@dataclass(frozen=True)
class AgentDecision:
    action: str
    message: str
    missing_field: str | None = None


FIELD_LABELS = {
    "origin": "出发地",
    "destination": "目的地",
    "start_date": "出发日期",
    "end_date": "返回日期",
    "traveler_count": "出行人数",
    "budget": "总预算",
}
