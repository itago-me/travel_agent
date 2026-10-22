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
from .booking import (
    CancelMockBookingRequest,
    CancelMockBookingResult,
    CreateMockBookingRequest,
    CreateMockBookingResult,
)
from .repository import SQLiteRepository


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
            description=(
                "查询指定路线、日期和人数的交通选项。"
                "不填 mode 会一次返回全部交通方式（火车/飞机/汽车），"
                "因此同一路线同一天只需调用一次——不要为了比较不同方式而"
                "分别调用。往返行程需要两次调用（去程和返程各一次）。"
            ),
            args_schema=TransportSearchRequest,
        ),
        StructuredTool.from_function(
            coroutine=search_hotel,
            name="search_hotel_options",
            description=(
                "查询指定城市和入住日期的酒店选项。rooms 是房间数不是人数。"
                "同一城市同一入住区间只需调用一次。"
            ),
            args_schema=HotelSearchRequest,
        ),
        StructuredTool.from_function(
            coroutine=search_attractions,
            name="search_attractions",
            description=(
                "查询指定城市和游览日期的景点选项。"
                "同一城市一次查询即可覆盖多个日期，不要按天分别调用。"
            ),
            args_schema=AttractionSearchRequest,
        ),
    ]


def create_sensitive_booking_tools(database: str | Path) -> list[StructuredTool]:
    """Build booking tools that must be paired with HITL before Agent exposure."""
    repository = SQLiteRepository(database)

    async def create_booking(**kwargs):
        request = CreateMockBookingRequest.model_validate(kwargs)
        booking = repository.create_mock_booking(
            request.consultation_id,
            request.itinerary_version_id,
            total_amount=request.total_amount,
            idempotency_key=request.idempotency_key,
        )
        return CreateMockBookingResult(
            booking_id=booking.id,
            consultation_id=booking.consultation_id,
            itinerary_version_id=booking.itinerary_version_id,
            total_amount=booking.total_amount,
            status=booking.status.value,
            idempotency_key=booking.idempotency_key,
        ).model_dump(mode="json")

    async def cancel_booking(**kwargs):
        request = CancelMockBookingRequest.model_validate(kwargs)
        booking = repository.cancel_mock_booking(
            request.consultation_id,
            request.booking_id,
            reason=request.reason,
            idempotency_key=request.idempotency_key,
        )
        return CancelMockBookingResult(
            booking_id=booking.id,
            consultation_id=booking.consultation_id,
            status=booking.status.value,
            cancellation_reason=booking.cancellation_reason or "",
            idempotency_key=booking.cancellation_idempotency_key or "",
        ).model_dump(mode="json")

    metadata = {"permission": "SENSITIVE", "requires_human_approval": True}
    return [
        StructuredTool.from_function(
            coroutine=create_booking,
            name="create_mock_booking",
            description="为已确认的旅行方案创建模拟预订。执行前必须获得人工审批。",
            args_schema=CreateMockBookingRequest,
            metadata=metadata,
        ),
        StructuredTool.from_function(
            coroutine=cancel_booking,
            name="cancel_mock_booking",
            description="取消已有模拟预订。执行前必须获得人工审批。",
            args_schema=CancelMockBookingRequest,
            metadata=metadata,
        ),
    ]
