from __future__ import annotations

from pathlib import Path

from .flyai_client import FlyAIClient
from .flyai_parsers import (
    parse_attraction_response,
    parse_hotel_response,
    parse_transport_response,
)
from .providers import (
    AttractionSearchRequest,
    AttractionSearchResult,
    HotelSearchRequest,
    HotelSearchResult,
    TransportMode,
    TransportSearchRequest,
    TransportSearchResult,
)


class FlyAIHotelProvider:
    def __init__(self, client: FlyAIClient | None = None):
        self.client = client or FlyAIClient()

    async def search(self, request: HotelSearchRequest) -> HotelSearchResult:
        payload = await self.client.search_hotel(request, sort="price_asc")
        return parse_hotel_response(payload, city=request.city)


class FlyAITransportProvider:
    def __init__(self, client: FlyAIClient | None = None):
        self.client = client or FlyAIClient()

    async def search(self, request: TransportSearchRequest) -> TransportSearchResult:
        mode = request.mode or TransportMode.TRAIN
        if mode == TransportMode.TRAIN:
            payload = await self.client.search_train(request, sort_type=3)
        elif mode == TransportMode.FLIGHT:
            payload = await self.client.search_flight(request, sort_type=3)
        else:
            raise ValueError(f"FlyAI transport provider does not support mode: {mode}")
        return parse_transport_response(payload, mode=mode)


class FlyAIAttractionProvider:
    def __init__(self, client: FlyAIClient | None = None):
        self.client = client or FlyAIClient()

    async def search(self, request: AttractionSearchRequest) -> AttractionSearchResult:
        payload = await self.client.search_poi(request)
        return parse_attraction_response(
            payload, city=request.city, visit_date=request.visit_date
        )
