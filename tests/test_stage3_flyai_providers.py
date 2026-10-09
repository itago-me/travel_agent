import asyncio

from travel_agent.flyai_providers import (
    FlyAIAttractionProvider,
    FlyAIHotelProvider,
    FlyAITransportProvider,
)
from travel_agent.providers import (
    AttractionSearchRequest,
    HotelSearchRequest,
    TransportMode,
    TransportSearchRequest,
)


class FakeFlyAIClient:
    async def search_hotel(self, request, **filters):
        assert request.city == "杭州"
        assert filters == {"sort": "price_asc"}
        return {
            "status": 0,
            "data": {"itemList": [{"shId": "hotel-1", "name": "测试酒店", "price": "¥200"}]},
        }

    async def search_train(self, request, **filters):
        assert filters == {"sort_type": 3}
        return {
            "status": 0,
            "data": {"itemList": [{
                "adultPrice": "¥100",
                "journeys": [{"segments": [{
                    "depCityName": "北京", "arrCityName": "杭州",
                    "depDateTime": "2026-10-15 08:00:00",
                    "arrDateTime": "2026-10-15 10:00:00",
                    "duration": "120分钟", "marketingTransportNo": "G1",
                }]}],
            }]},
        }

    async def search_flight(self, request, **filters):
        return {"status": 0, "data": {"itemList": []}}

    async def search_poi(self, request, **filters):
        assert request.city == "杭州"
        return {"status": 0, "data": {"itemList": []}}


def test_flyai_hotel_provider_returns_standard_result():
    result = asyncio.run(
        FlyAIHotelProvider(FakeFlyAIClient()).search(
            HotelSearchRequest(city="杭州", check_in="2026-10-15", check_out="2026-10-17", rooms=1)
        )
    )
    assert result.provider == "flyai_fliggy"
    assert result.options[0].price_per_night == 200


def test_flyai_transport_provider_routes_train_and_rejects_bus():
    provider = FlyAITransportProvider(FakeFlyAIClient())
    result = asyncio.run(
        provider.search(
            TransportSearchRequest(origin="北京", destination="杭州", departure_date="2026-10-15", traveler_count=1, mode=TransportMode.TRAIN)
        )
    )
    assert result.options[0].vehicle_type == "high_speed_train"

    import pytest
    with pytest.raises(ValueError, match="does not support mode"):
        asyncio.run(
            provider.search(
                TransportSearchRequest(origin="北京", destination="杭州", departure_date="2026-10-15", traveler_count=1, mode=TransportMode.BUS)
            )
        )


def test_flyai_attraction_provider_returns_standard_empty_result():
    result = asyncio.run(
        FlyAIAttractionProvider(FakeFlyAIClient()).search(
            AttractionSearchRequest(city="杭州", visit_date="2026-10-15")
        )
    )
    assert result.provider == "flyai_fliggy"
    assert result.options == []
