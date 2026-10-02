from datetime import date

from travel_agent.calculators import (
    BudgetCalculationRequest,
    ConflictDetectionRequest,
    ItineraryScoreRequest,
    calculate_trip_budget,
    detect_itinerary_conflicts,
    score_trip_options,
)
from travel_agent.providers import (
    AttractionOption,
    HotelOption,
    TransportMode,
    TransportOption,
)


def sample_options():
    transport = TransportOption(
        provider="mock_transport", option_id="train-001", mode=TransportMode.TRAIN,
        origin="北京", destination="杭州", departure_date=date(2026, 10, 1),
        departure_time="08:00", arrival_time="13:30", duration_minutes=330,
        total_price=1800,
    )
    hotel = HotelOption(
        provider="mock_hotel", option_id="hotel-001", name="西湖假日酒店", city="杭州",
        price_per_night=680, rating=4.6, available_rooms=5,
    )
    attractions = [
        AttractionOption(
            provider="mock_attraction", option_id="a-001", name="西湖", city="杭州",
            visit_date=date(2026, 10, 2), duration_minutes=180, ticket_price=0,
        ),
        AttractionOption(
            provider="mock_attraction", option_id="a-002", name="灵隐寺", city="杭州",
            visit_date=date(2026, 10, 3), duration_minutes=150, ticket_price=75,
        ),
    ]
    return transport, hotel, attractions


def test_budget_calculation_multiplies_hotel_by_nights():
    transport, hotel, attractions = sample_options()
    result = calculate_trip_budget(
        BudgetCalculationRequest(
            transport=transport, hotel=hotel, attractions=attractions,
            check_in=date(2026, 10, 1), check_out=date(2026, 10, 4), budget_limit=8000,
        )
    )

    assert result.transport_total == 1800
    assert result.hotel_total == 2040
    assert result.attractions_total == 75
    assert result.total == 3915
    assert result.within_budget is True


def test_conflict_detection_reports_overlapping_attraction_visits():
    transport, hotel, attractions = sample_options()
    overlapping = attractions + [attractions[0].model_copy(update={"option_id": "a-003", "name": "重复景点"})]

    result = detect_itinerary_conflicts(
        ConflictDetectionRequest(transport=transport, attractions=overlapping)
    )

    assert result.has_conflicts is True
    assert len(result.conflicts) == 1
    assert "重复景点" in result.conflicts[0]


def test_score_is_deterministic_and_penalizes_budget_overrun():
    transport, hotel, attractions = sample_options()
    result = score_trip_options(
        ItineraryScoreRequest(
            transport=transport, hotel=hotel, attractions=attractions,
            total_budget=3915, budget_limit=8000, conflict_count=0,
        )
    )
    over_budget = score_trip_options(
        ItineraryScoreRequest(
            transport=transport, hotel=hotel, attractions=attractions,
            total_budget=10000, budget_limit=8000, conflict_count=0,
        )
    )

    assert result.score == score_trip_options(
        ItineraryScoreRequest(
            transport=transport, hotel=hotel, attractions=attractions,
            total_budget=3915, budget_limit=8000, conflict_count=0,
        )
    ).score
    assert result.score > over_budget.score
