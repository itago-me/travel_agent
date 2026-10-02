from datetime import date, datetime, timedelta

from pydantic import Field

from .providers import (
    AttractionOption,
    HotelOption,
    ProviderModel,
    TransportOption,
)


class BudgetCalculationRequest(ProviderModel):
    transport: TransportOption
    hotel: HotelOption
    attractions: list[AttractionOption]
    check_in: date
    check_out: date
    budget_limit: float = Field(ge=0)


class BudgetCalculationResult(ProviderModel):
    transport_total: float
    hotel_total: float
    attractions_total: float
    total: float
    budget_limit: float
    within_budget: bool
    nights: int


class ConflictDetectionRequest(ProviderModel):
    transport: TransportOption
    attractions: list[AttractionOption]


class ConflictDetectionResult(ProviderModel):
    has_conflicts: bool
    conflicts: list[str]


class ItineraryScoreRequest(ProviderModel):
    transport: TransportOption
    hotel: HotelOption
    attractions: list[AttractionOption]
    total_budget: float = Field(ge=0)
    budget_limit: float = Field(ge=0)
    conflict_count: int = Field(ge=0)


class ItineraryScoreResult(ProviderModel):
    score: float
    reasons: list[str]


def calculate_trip_budget(request: BudgetCalculationRequest) -> BudgetCalculationResult:
    nights = (request.check_out - request.check_in).days
    transport_total = request.transport.total_price
    hotel_total = request.hotel.price_per_night * nights
    attractions_total = sum(item.ticket_price for item in request.attractions)
    total = transport_total + hotel_total + attractions_total
    return BudgetCalculationResult(
        transport_total=transport_total,
        hotel_total=hotel_total,
        attractions_total=attractions_total,
        total=total,
        budget_limit=request.budget_limit,
        within_budget=total <= request.budget_limit,
        nights=nights,
    )


def detect_itinerary_conflicts(
    request: ConflictDetectionRequest,
) -> ConflictDetectionResult:
    conflicts: list[str] = []
    seen_dates: dict[date, AttractionOption] = {}
    for attraction in request.attractions:
        previous = seen_dates.get(attraction.visit_date)
        if previous is not None:
            conflicts.append(
                f"{previous.name} 与 {attraction.name} 在 {attraction.visit_date.isoformat()} 重复安排"
            )
        else:
            seen_dates[attraction.visit_date] = attraction

    arrival = datetime.combine(request.transport.departure_date, request.transport.arrival_time)
    for attraction in request.attractions:
        if attraction.visit_date == request.transport.departure_date:
            conflicts.append(
                f"交通抵达时间 {arrival.time().isoformat(timespec='minutes')} 与 {attraction.name} 当日安排存在风险"
            )
    return ConflictDetectionResult(has_conflicts=bool(conflicts), conflicts=conflicts)


def score_trip_options(request: ItineraryScoreRequest) -> ItineraryScoreResult:
    score = 100.0
    reasons: list[str] = []
    if request.total_budget > request.budget_limit:
        score -= min(40.0, (request.total_budget - request.budget_limit) / request.budget_limit * 100)
        reasons.append("超出预算")
    else:
        reasons.append("预算内")
    score += request.hotel.rating * 2
    score -= min(10.0, request.transport.duration_minutes / 60)
    if request.conflict_count:
        score -= request.conflict_count * 20
        reasons.append("存在时间冲突")
    score = max(0.0, min(100.0, round(score, 2)))
    return ItineraryScoreResult(score=score, reasons=reasons)
