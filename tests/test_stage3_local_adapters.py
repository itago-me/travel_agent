import asyncio
from datetime import date

from travel_agent.providers import (
    AttractionSearchRequest,
    HotelSearchRequest,
    TransportMode,
    TransportSearchRequest,
)
from travel_agent.sqlite_providers import (
    SQLiteAttractionProvider,
    SQLiteHotelProvider,
    SQLiteTransportProvider,
    initialize_travel_data,
)


def test_sqlite_transport_provider_reads_persisted_options(tmp_path):
    database = tmp_path / "travel.db"
    initialize_travel_data(database)

    result = asyncio.run(
        SQLiteTransportProvider(database).search(
            TransportSearchRequest(
                origin="北京",
                destination="杭州",
                departure_date=date(2026, 10, 1),
                traveler_count=2,
                mode=TransportMode.TRAIN,
            )
        )
    )

    assert result.provider == "sqlite_transport"
    assert [option.option_id for option in result.options] == [
        "train-bj-hz-001",
        "train-bj-hz-002",
    ]


def test_sqlite_hotel_and_attraction_providers_return_standard_schemas(tmp_path):
    database = tmp_path / "travel.db"
    initialize_travel_data(database)

    hotel_result = asyncio.run(
        SQLiteHotelProvider(database).search(
            HotelSearchRequest(
                city="杭州",
                check_in=date(2026, 10, 1),
                check_out=date(2026, 10, 4),
                rooms=1,
            )
        )
    )
    attraction_result = asyncio.run(
        SQLiteAttractionProvider(database).search(
            AttractionSearchRequest(city="杭州", visit_date=date(2026, 10, 2))
        )
    )

    assert hotel_result.provider == "sqlite_hotel"
    assert hotel_result.options[0].city == "杭州"
    assert attraction_result.provider == "sqlite_attraction"
    assert all(option.visit_date == date(2026, 10, 2) for option in attraction_result.options)
