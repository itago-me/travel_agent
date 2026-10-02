from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from .mock_providers import (
    MockAttractionProvider,
    MockHotelProvider,
    MockTransportProvider,
)
from .calculators import (
    BudgetCalculationRequest,
    ConflictDetectionRequest,
    ItineraryScoreRequest,
    calculate_trip_budget,
    detect_itinerary_conflicts,
    score_trip_options,
)
from .providers import (
    AttractionSearchRequest,
    HotelSearchRequest,
    TransportSearchRequest,
)


class ToolPermission(StrEnum):
    READ = "READ"
    COMPUTE = "COMPUTE"
    WRITE = "WRITE"
    SENSITIVE = "SENSITIVE"


class ToolGatewayError(Exception):
    pass


class UnknownToolError(ToolGatewayError):
    pass


class PermissionDeniedError(ToolGatewayError):
    pass


@dataclass(frozen=True)
class ToolExecutionContext:
    consultation_id: str
    thread_id: str
    allowed_permissions: set[ToolPermission]


@dataclass(frozen=True)
class ToolAuditEvent:
    consultation_id: str
    thread_id: str
    tool_name: str
    permission: ToolPermission
    call_number: int
    outcome: str
    error_type: str | None = None


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_name: str
    call_number: int
    output: dict[str, Any]


ToolHandler = Callable[[BaseModel], Awaitable[BaseModel]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    permission: ToolPermission
    input_model: type[BaseModel] = field(repr=False)
    handler: ToolHandler = field(repr=False)


class ToolGateway:
    def __init__(
        self,
        audit_sink: Callable[[ToolAuditEvent], None] | None = None,
    ) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._call_counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
        self._audit_sink = audit_sink or (lambda event: None)

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ValueError(f"Tool already registered: {definition.name}")
        self._tools[definition.name] = definition

    def list_tools(self) -> list[ToolDefinition]:
        return sorted(self._tools.values(), key=lambda tool: tool.name)

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        definition = self._tools.get(tool_name)
        if definition is None:
            raise UnknownToolError(f"Unknown tool: {tool_name}")

        key = (context.consultation_id, context.thread_id, tool_name)
        call_number = self._call_counts[key] + 1
        if definition.permission not in context.allowed_permissions:
            self._audit(
                definition,
                context,
                call_number,
                outcome="DENIED",
                error_type="PermissionDeniedError",
            )
            raise PermissionDeniedError(
                f"Permission {definition.permission} required for {tool_name}"
            )

        try:
            request = definition.input_model.model_validate(arguments)
        except Exception as error:
            self._audit(
                definition,
                context,
                call_number,
                outcome="REJECTED",
                error_type=type(error).__name__,
            )
            raise

        self._call_counts[key] = call_number
        try:
            response = await definition.handler(request)
        except Exception as error:
            self._audit(
                definition,
                context,
                call_number,
                outcome="FAILED",
                error_type=type(error).__name__,
            )
            raise

        self._audit(definition, context, call_number, outcome="SUCCEEDED")
        return ToolExecutionResult(
            tool_name=tool_name,
            call_number=call_number,
            output=response.model_dump(mode="json"),
        )

    def _audit(
        self,
        definition: ToolDefinition,
        context: ToolExecutionContext,
        call_number: int,
        *,
        outcome: str,
        error_type: str | None = None,
    ) -> None:
        self._audit_sink(
            ToolAuditEvent(
                consultation_id=context.consultation_id,
                thread_id=context.thread_id,
                tool_name=definition.name,
                permission=definition.permission,
                call_number=call_number,
                outcome=outcome,
                error_type=error_type,
            )
        )


def create_mock_tool_gateway(
    audit_sink: Callable[[ToolAuditEvent], None] | None = None,
) -> ToolGateway:
    transport = MockTransportProvider()
    hotel = MockHotelProvider()
    attraction = MockAttractionProvider()
    gateway = ToolGateway(audit_sink=audit_sink)
    gateway.register(
        ToolDefinition(
            name="search_transport_options",
            description="查询符合出发地、目的地、日期和人数的交通选项",
            permission=ToolPermission.READ,
            input_model=TransportSearchRequest,
            handler=transport.search,
        )
    )
    gateway.register(
        ToolDefinition(
            name="search_hotel_options",
            description="查询城市、入住日期和房间数匹配的酒店选项",
            permission=ToolPermission.READ,
            input_model=HotelSearchRequest,
            handler=hotel.search,
        )
    )
    gateway.register(
        ToolDefinition(
            name="search_attractions",
            description="查询指定城市和游览日期的景点选项",
            permission=ToolPermission.READ,
            input_model=AttractionSearchRequest,
            handler=attraction.search,
        )
    )
    return gateway


def create_planning_tool_gateway(
    audit_sink: Callable[[ToolAuditEvent], None] | None = None,
) -> ToolGateway:
    gateway = create_mock_tool_gateway(audit_sink=audit_sink)

    async def budget_handler(request: BudgetCalculationRequest):
        return calculate_trip_budget(request)

    async def conflict_handler(request: ConflictDetectionRequest):
        return detect_itinerary_conflicts(request)

    async def score_handler(request: ItineraryScoreRequest):
        return score_trip_options(request)

    gateway.register(
        ToolDefinition(
            name="calculate_trip_budget",
            description="根据交通、酒店和景点选项计算总预算",
            permission=ToolPermission.COMPUTE,
            input_model=BudgetCalculationRequest,
            handler=budget_handler,
        )
    )
    gateway.register(
        ToolDefinition(
            name="detect_itinerary_conflicts",
            description="检测交通与景点安排的时间冲突",
            permission=ToolPermission.COMPUTE,
            input_model=ConflictDetectionRequest,
            handler=conflict_handler,
        )
    )
    gateway.register(
        ToolDefinition(
            name="score_trip_options",
            description="根据预算、评分、耗时和冲突为候选方案打分",
            permission=ToolPermission.COMPUTE,
            input_model=ItineraryScoreRequest,
            handler=score_handler,
        )
    )
    return gateway
