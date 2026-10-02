from datetime import date

import pytest
from pydantic import ValidationError

from travel_agent.providers import (
    AttractionOption,
    AttractionSearchRequest,
    HotelOption,
    HotelSearchRequest,
    TransportMode,
    TransportOption,
    TransportSearchRequest,
)


def test_transport_request_and_option_have_normalized_fields():
    request = TransportSearchRequest(
        origin="北京",
        destination="杭州",
        departure_date=date(2026, 10, 1),
        traveler_count=2,
        mode=TransportMode.TRAIN,
    )
    option = TransportOption(
        provider="mock_transport",
        option_id="train-001",
        mode=TransportMode.TRAIN,
        origin=request.origin,
        destination=request.destination,
        departure_date=request.departure_date,
        departure_time="08:00",
        arrival_time="13:30",
        duration_minutes=330,
        total_price=1800,
    )

    assert request.traveler_count == 2
    assert option.total_price == 1800
    assert option.model_dump(mode="json")["departure_date"] == "2026-10-01"


def test_hotel_and_attraction_options_share_provider_identity_fields():
    hotel_request = HotelSearchRequest(
        city="杭州", check_in=date(2026, 10, 1), check_out=date(2026, 10, 4), rooms=1
    )
    hotel = HotelOption(
        provider="mock_hotel",
        option_id="hotel-001",
        name="西湖假日酒店",
        city=hotel_request.city,
        price_per_night=680,
        rating=4.6,
        available_rooms=5,
    )
    attraction_request = AttractionSearchRequest(city="杭州", visit_date=date(2026, 10, 2))
    attraction = AttractionOption(
        provider="mock_attraction",
        option_id="attraction-001",
        name="西湖",
        city=attraction_request.city,
        visit_date=attraction_request.visit_date,
        duration_minutes=180,
        ticket_price=0,
    )

    assert hotel.provider == "mock_hotel"
    assert attraction.provider == "mock_attraction"
    assert hotel.city == attraction.city


def test_requests_reject_invalid_business_values():
    with pytest.raises(ValidationError):
        HotelSearchRequest(city="杭州", check_in=date(2026, 10, 4), check_out=date(2026, 10, 1))
    with pytest.raises(ValidationError):
        TransportSearchRequest(
            origin="北京",
            destination="杭州",
            departure_date=date(2026, 10, 1),
            traveler_count=0,
        )
