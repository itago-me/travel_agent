import asyncio

import pytest

from travel_agent.langchain_tools import create_travel_tools
from travel_agent.provider_factory import ProviderSettings, create_provider_bundle
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
)


def test_provider_settings_default_to_sqlite(monkeypatch):
    for name in (
        "TRAVEL_AGENT_HOTEL_PROVIDER",
        "TRAVEL_AGENT_TRANSPORT_PROVIDER",
        "TRAVEL_AGENT_ATTRACTION_PROVIDER",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = ProviderSettings.from_env()

    assert settings.hotel == "sqlite"
    assert settings.transport == "sqlite"
    assert settings.attraction == "sqlite"


def test_factory_selects_sqlite_providers_by_configuration(tmp_path):
    bundle = create_provider_bundle(
        tmp_path / "travel.db",
        ProviderSettings(hotel="sqlite", transport="sqlite", attraction="sqlite"),
    )

    assert isinstance(bundle.hotel, SQLiteHotelProvider)
    assert isinstance(bundle.transport, SQLiteTransportProvider)
    assert isinstance(bundle.attraction, SQLiteAttractionProvider)


def test_factory_rejects_unknown_provider_name(tmp_path):
    with pytest.raises(ValueError, match="Unsupported hotel provider"):
        create_provider_bundle(
            tmp_path / "travel.db",
            ProviderSettings(hotel="unknown", transport="sqlite", attraction="sqlite"),
        )


def test_tools_keep_names_when_provider_mode_is_sqlite(tmp_path):
    tools = create_travel_tools(tmp_path / "travel.db", ProviderSettings())

    assert [tool.name for tool in tools] == [
        "search_transport_options",
        "search_hotel_options",
        "search_attractions",
    ]
