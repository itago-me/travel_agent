from datetime import date, time

from .providers import (
    AttractionOption,
    AttractionSearchRequest,
    AttractionSearchResult,
    HotelOption,
    HotelSearchRequest,
    HotelSearchResult,
    TransportMode,
    TransportOption,
    TransportSearchRequest,
    TransportSearchResult,
)


class MockTransportProvider:
    provider_name = "mock_transport"

    async def search(self, request: TransportSearchRequest) -> TransportSearchResult:
        options = []
        if (request.origin, request.destination) == ("北京", "杭州"):
            options = [
                TransportOption(
                    provider=self.provider_name,
                    option_id="train-bj-hz-001",
                    mode=TransportMode.TRAIN,
                    origin=request.origin,
                    destination=request.destination,
                    departure_date=request.departure_date,
                    departure_time=time(8, 0),
                    arrival_time=time(13, 30),
                    duration_minutes=330,
                    total_price=1800 * request.traveler_count / 2,
                ),
                TransportOption(
                    provider=self.provider_name,
                    option_id="train-bj-hz-002",
                    mode=TransportMode.TRAIN,
                    origin=request.origin,
                    destination=request.destination,
                    departure_date=request.departure_date,
                    departure_time=time(14, 0),
                    arrival_time=time(19, 45),
                    duration_minutes=345,
                    total_price=1500 * request.traveler_count / 2,
                ),
            ]
        if request.mode is not None:
            options = [option for option in options if option.mode == request.mode]
        return TransportSearchResult(provider=self.provider_name, options=options)


class MockHotelProvider:
    provider_name = "mock_hotel"

    async def search(self, request: HotelSearchRequest) -> HotelSearchResult:
        nights = (request.check_out - request.check_in).days
        options = []
        if request.city == "杭州":
            options = [
                HotelOption(
                    provider=self.provider_name,
                    option_id="hotel-hz-001",
                    name="西湖假日酒店",
                    city=request.city,
                    price_per_night=680 * request.rooms,
                    rating=4.6,
                    available_rooms=5,
                ),
                HotelOption(
                    provider=self.provider_name,
                    option_id="hotel-hz-002",
                    name="钱塘精选酒店",
                    city=request.city,
                    price_per_night=420 * request.rooms,
                    rating=4.1,
                    available_rooms=8,
                ),
            ]
        return HotelSearchResult(provider=self.provider_name, options=options)


class MockAttractionProvider:
    provider_name = "mock_attraction"

    async def search(self, request: AttractionSearchRequest) -> AttractionSearchResult:
        options = []
        if request.city == "杭州":
            options = [
                AttractionOption(
                    provider=self.provider_name,
                    option_id="attraction-hz-001",
                    name="西湖",
                    city=request.city,
                    visit_date=request.visit_date,
                    duration_minutes=180,
                    ticket_price=0,
                ),
                AttractionOption(
                    provider=self.provider_name,
                    option_id="attraction-hz-002",
                    name="灵隐寺",
                    city=request.city,
                    visit_date=request.visit_date,
                    duration_minutes=150,
                    ticket_price=75,
                ),
            ]
        return AttractionSearchResult(provider=self.provider_name, options=options)
