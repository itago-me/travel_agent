from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .flyai_client import FlyAIClient
from .flyai_providers import (
    FlyAIAttractionProvider,
    FlyAIHotelProvider,
    FlyAITransportProvider,
)
from .providers import AttractionProvider, HotelProvider, TransportProvider
from .sqlite_providers import (
    SQLiteAttractionProvider,
    SQLiteHotelProvider,
    SQLiteTransportProvider,
)


@dataclass(frozen=True)
class ProviderSettings:
    hotel: str = "sqlite"
    transport: str = "sqlite"
    attraction: str = "sqlite"

    @classmethod
    def from_env(cls) -> "ProviderSettings":
        return cls(
            hotel=os.getenv("TRAVEL_AGENT_HOTEL_PROVIDER", "sqlite").lower(),
            transport=os.getenv("TRAVEL_AGENT_TRANSPORT_PROVIDER", "sqlite").lower(),
            attraction=os.getenv("TRAVEL_AGENT_ATTRACTION_PROVIDER", "sqlite").lower(),
        )


@dataclass(frozen=True)
class ProviderBundle:
    hotel: HotelProvider
    transport: TransportProvider
    attraction: AttractionProvider


def create_provider_bundle(
    database: str | Path,
    settings: ProviderSettings | None = None,
    *,
    flyai_client: FlyAIClient | None = None,
) -> ProviderBundle:
    settings = settings or ProviderSettings.from_env()
    return ProviderBundle(
        hotel=_create_one("hotel", settings.hotel, database, flyai_client),
        transport=_create_one("transport", settings.transport, database, flyai_client),
        attraction=_create_one("attraction", settings.attraction, database, flyai_client),
    )


def _create_one(
    kind: str,
    provider: str,
    database: str | Path,
    flyai_client: FlyAIClient | None,
):
    if provider == "sqlite":
        classes = {
            "hotel": SQLiteHotelProvider,
            "transport": SQLiteTransportProvider,
            "attraction": SQLiteAttractionProvider,
        }
        return classes[kind](database)
    if provider == "flyai":
        classes = {
            "hotel": FlyAIHotelProvider,
            "transport": FlyAITransportProvider,
            "attraction": FlyAIAttractionProvider,
        }
        return classes[kind](flyai_client)
    raise ValueError(f"Unsupported {kind} provider: {provider}")
