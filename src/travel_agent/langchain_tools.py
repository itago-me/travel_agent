"""LangChain tools backed by normalized travel provider contracts."""

from pathlib import Path

from langchain_core.tools import StructuredTool

from .flyai_client import FlyAIClient
from .provider_factory import ProviderSettings, create_provider_bundle
from .providers import (
    AttractionSearchRequest,
    HotelSearchRequest,
    TransportSearchRequest,
)
def create_travel_tools(
    database: str | Path,
    provider_settings: ProviderSettings | None = None,
    *,
    flyai_client: FlyAIClient | None = None,
) -> list[StructuredTool]:
    """Build tools the Agent can bind; each tool returns JSON-ready data."""
    providers = create_provider_bundle(
        database,
        provider_settings,
        flyai_client=flyai_client,
    )
    transport = providers.transport
    hotel = providers.hotel
    attraction = providers.attraction

    async def search_transport(**kwargs):
        result = await transport.search(TransportSearchRequest.model_validate(kwargs))
        return result.model_dump(mode="json")

    async def search_hotel(**kwargs):
        result = await hotel.search(HotelSearchRequest.model_validate(kwargs))
        return result.model_dump(mode="json")

    async def search_attractions(**kwargs):
        result = await attraction.search(AttractionSearchRequest.model_validate(kwargs))
        return result.model_dump(mode="json")

    return [
        StructuredTool.from_function(
            coroutine=search_transport,
            name="search_transport_options",
            description="查询指定路线、日期和人数的交通选项。",
            args_schema=TransportSearchRequest,
        ),
        StructuredTool.from_function(
            coroutine=search_hotel,
            name="search_hotel_options",
            description="查询指定城市和入住日期的酒店选项。",
            args_schema=HotelSearchRequest,
        ),
        StructuredTool.from_function(
            coroutine=search_attractions,
            name="search_attractions",
            description="查询指定城市和游览日期的景点选项。",
            args_schema=AttractionSearchRequest,
        ),
    ]
