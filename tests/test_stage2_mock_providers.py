import asyncio
from datetime import date

from travel_agent.mock_providers import (
    MockAttractionProvider,
    MockHotelProvider,
    MockTransportProvider,
)
from travel_agent.providers import (
    AttractionSearchRequest,
    HotelSearchRequest,
    TransportMode,
    TransportSearchRequest,
)


def test_mock_transport_returns_repeatable_options_for_a_route():
    provider = MockTransportProvider()
    request = TransportSearchRequest(
        origin="北京",
        destination="杭州",
        departure_date=date(2026, 10, 1),
        traveler_count=2,
        mode=TransportMode.TRAIN,
    )

    first = asyncio.run(provider.search(request))
    second = asyncio.run(provider.search(request))

    assert first == second
    assert first.provider == "mock_transport"
    assert len(first.options) == 2
    assert {option.mode for option in first.options} == {TransportMode.TRAIN}
    assert all(option.total_price > 0 for option in first.options)


def test_mock_hotel_returns_options_with_prices_for_stay_length():
    provider = MockHotelProvider()
    request = HotelSearchRequest(
        city="杭州", check_in=date(2026, 10, 1), check_out=date(2026, 10, 4), rooms=1
    )

    result = asyncio.run(provider.search(request))

    assert result.provider == "mock_hotel"
    assert len(result.options) == 2
    assert {option.city for option in result.options} == {"杭州"}
    assert all(option.price_per_night > 0 for option in result.options)


def test_mock_attraction_returns_city_options_for_visit_date():
    provider = MockAttractionProvider()
    request = AttractionSearchRequest(city="杭州", visit_date=date(2026, 10, 2))

    result = asyncio.run(provider.search(request))

    assert result.provider == "mock_attraction"
    assert len(result.options) == 2
    assert all(option.city == "杭州" for option in result.options)
    assert all(option.visit_date == date(2026, 10, 2) for option in result.options)


def test_mock_providers_return_no_options_for_unknown_route_or_city():
    transport = asyncio.run(
        MockTransportProvider().search(
            TransportSearchRequest(
                origin="广州",
                destination="昆明",
                departure_date=date(2026, 10, 1),
                traveler_count=1,
            )
        )
    )
    hotels = asyncio.run(
        MockHotelProvider().search(
            HotelSearchRequest(
                city="不存在城市",
                check_in=date(2026, 10, 1),
                check_out=date(2026, 10, 2),
                rooms=1,
            )
        )
    )

    assert transport.options == []
    assert hotels.options == []
