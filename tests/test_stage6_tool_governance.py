"""阶段 6.2 A/B：工具执行审计与调用预算。

A（审计）：`create_mock_booking` / `cancel_mock_booking` 是唯一写资金的敏感
操作，原先却是唯一没有审计的写操作——审计链只能证明"顾问批准过"，证明不了
"批准后实际执行了什么"。工具执行成功/失败同样没有留痕。

B（上限）：全库没有 recursion_limit，也没有工具调用计数，模型陷入循环时
没有任何停止条件。
"""

import asyncio
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from travel_agent.agent import AgentSettings, create_agent_app
from travel_agent.application import create_consultation_service
from travel_agent.domain import AuditEvent, BookingStatus
from travel_agent.repository import SQLiteRepository
from travel_agent.tool_governance import ToolGovernanceMiddleware


def make_service(tmp_path):
    service = create_consultation_service(tmp_path / "travel.db")
    consultation = service.create_consultation("c1", "客户", "去杭州")
    return service, consultation


def seed_version(service, consultation_id, total=2790):
    version_id, _ = service.repository.save_itinerary_version(
        consultation_id, {"budget": {"total": total}}, [], {}
    )
    service.repository.select_itinerary_version(consultation_id, version_id)
    return version_id


def event_types(service, consultation_id):
    return [
        event.event_type
        for event in service.repository.list_audit_events(consultation_id)
    ]


# --- A. 订单审计 ------------------------------------------------------------


def test_creating_a_booking_records_an_audit_event(tmp_path):
    service, consultation = make_service(tmp_path)
    version_id = seed_version(service, consultation.id)

    booking = service.repository.create_mock_booking(
        consultation.id, version_id, total_amount=2790, idempotency_key="k1"
    )

    events = [
        event
        for event in service.repository.list_audit_events(consultation.id)
        if event.event_type == "BOOKING_CREATED"
    ]
    assert len(events) == 1
    details = events[0].details
    assert details["booking_id"] == booking.id
    assert details["itinerary_version_id"] == version_id
    assert details["total_amount"] == 2790
    assert details["idempotency_key"] == "k1"


def test_idempotent_replay_does_not_record_a_second_booking_event(tmp_path):
    """幂等重放返回原订单，不能让人以为又建了一单。"""
    service, consultation = make_service(tmp_path)
    version_id = seed_version(service, consultation.id)

    service.repository.create_mock_booking(
        consultation.id, version_id, total_amount=2790, idempotency_key="k1"
    )
    service.repository.create_mock_booking(
        consultation.id, version_id, total_amount=2790, idempotency_key="k1"
    )

    assert event_types(service, consultation.id).count("BOOKING_CREATED") == 1


def test_failed_booking_records_no_audit_event(tmp_path):
    """校验失败时不能留下"已创建"的假记录。"""
    service, consultation = make_service(tmp_path)
    version_id = seed_version(service, consultation.id)

    with pytest.raises(ValueError):
        service.repository.create_mock_booking(
            consultation.id, version_id, total_amount=0, idempotency_key="k1"
        )

    assert "BOOKING_CREATED" not in event_types(service, consultation.id)


def test_cancelling_a_booking_records_an_audit_event(tmp_path):
    service, consultation = make_service(tmp_path)
    version_id = seed_version(service, consultation.id)
    booking = service.repository.create_mock_booking(
        consultation.id, version_id, total_amount=2790, idempotency_key="k1"
    )

    service.repository.cancel_mock_booking(
        consultation.id, booking.id, reason="客户改期", idempotency_key="c1"
    )

    events = [
        event
        for event in service.repository.list_audit_events(consultation.id)
        if event.event_type == "BOOKING_CANCELLED"
    ]
    assert len(events) == 1
    assert events[0].details["booking_id"] == booking.id
    assert events[0].details["cancellation_reason"] == "客户改期"


def test_repeated_cancel_does_not_record_a_second_event(tmp_path):
    service, consultation = make_service(tmp_path)
    version_id = seed_version(service, consultation.id)
    booking = service.repository.create_mock_booking(
        consultation.id, version_id, total_amount=2790, idempotency_key="k1"
    )
    for _ in range(2):
        service.repository.cancel_mock_booking(
            consultation.id, booking.id, reason="客户改期", idempotency_key="c1"
        )

    assert event_types(service, consultation.id).count("BOOKING_CANCELLED") == 1


# --- A. 工具执行审计 --------------------------------------------------------


class ToolCallingModel(BaseChatModel):
    tool_name: str = "search_hotel_options"
    tool_args: dict[str, Any] = {}
    invocations: int = 0

    @property
    def _llm_type(self):
        return "offline-tool-calling"

    def bind_tools(self, tools, **kwargs):
        return self

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.invocations += 1
        if any(isinstance(message, ToolMessage) for message in messages):
            message = AIMessage(content="已完成查询")
        else:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": self.tool_name,
                        "args": self.tool_args,
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


def run_agent_turn(monkeypatch, database, consultation_id, model):
    """Run one real agent turn with governance middleware attached."""
    from travel_agent.langchain_tools import (
        create_sensitive_booking_tools,
        create_travel_tools,
    )

    monkeypatch.setattr("travel_agent.agent.create_chat_model", lambda _settings: model)
    settings = AgentSettings(
        model_mode="openai_compatible", model_name="demo", api_key=None, base_url=None
    )
    agent = create_agent_app(
        database,
        settings,
        tools=[
            *create_travel_tools(database),
            *create_sensitive_booking_tools(database),
        ],
        consultation_id=consultation_id,
    )
    return asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "查询"}]}))


def test_successful_tool_call_is_recorded(monkeypatch, tmp_path):
    service, consultation = make_service(tmp_path)
    model = ToolCallingModel(
        tool_name="search_hotel_options",
        tool_args={
            "city": "杭州",
            "check_in": "2026-10-01",
            "check_out": "2026-10-03",
            "rooms": 1,
        },
    )

    run_agent_turn(monkeypatch, service.repository.database_path, consultation.id, model)

    assert "TOOL_EXECUTED" in event_types(service, consultation.id)


def test_failed_tool_call_is_recorded_with_the_error(monkeypatch, tmp_path):
    service, consultation = make_service(tmp_path)
    # 缺少必填参数，工具内部校验会失败
    model = ToolCallingModel(tool_name="search_hotel_options", tool_args={})

    run_agent_turn(monkeypatch, service.repository.database_path, consultation.id, model)

    events = [
        event
        for event in service.repository.list_audit_events(consultation.id)
        if event.event_type == "TOOL_FAILED"
    ]
    assert events, "工具失败必须留下审计记录"
    assert events[0].details["tool_name"] == "search_hotel_options"


# --- B. 调用预算 ------------------------------------------------------------


def test_tool_execution_counter_counts_success_and_failure(tmp_path):
    service, consultation = make_service(tmp_path)
    repository = service.repository
    repository.add_audit_event(
        AuditEvent(consultation.id, "TOOL_EXECUTED", {"tool_name": "a"})
    )
    repository.add_audit_event(
        AuditEvent(consultation.id, "TOOL_FAILED", {"tool_name": "b"})
    )
    repository.add_audit_event(
        AuditEvent(consultation.id, "MESSAGE_APPENDED", {"role": "user"})
    )

    assert repository.count_tool_executions(consultation.id) == 2


def test_budget_blocks_further_calls_once_exhausted(tmp_path):
    service, consultation = make_service(tmp_path)
    repository = service.repository
    for _ in range(3):
        repository.add_audit_event(
            AuditEvent(consultation.id, "TOOL_EXECUTED", {"tool_name": "x"})
        )

    middleware = ToolGovernanceMiddleware(
        repository, consultation.id, max_calls=3
    )
    request = type(
        "Req",
        (),
        {
            "tool_call": {"name": "search_hotel_options", "args": {}, "id": "c1"},
            "tool": None,
        },
    )()

    called = False

    async def handler(_request):
        nonlocal called
        called = True
        return ToolMessage(content="ok", tool_call_id="c1")

    result = asyncio.run(middleware.awrap_tool_call(request, handler))

    assert called is False, "预算用尽后不得执行工具"
    assert result.status == "error"
    assert "预算已用尽" in result.content


def test_budget_allows_calls_below_the_limit(tmp_path):
    service, consultation = make_service(tmp_path)
    repository = service.repository
    middleware = ToolGovernanceMiddleware(repository, consultation.id, max_calls=3)
    request = type(
        "Req",
        (),
        {
            "tool_call": {"name": "search_hotel_options", "args": {}, "id": "c1"},
            "tool": None,
        },
    )()

    async def handler(_request):
        return ToolMessage(content="ok", tool_call_id="c1")

    result = asyncio.run(middleware.awrap_tool_call(request, handler))

    assert result.status != "error"
    assert repository.count_tool_executions(consultation.id) == 1


def test_agent_config_carries_a_recursion_limit():
    from travel_agent.service import _agent_config

    config = _agent_config("thread-1")

    assert config["configurable"]["thread_id"] == "thread-1"
    assert isinstance(config["recursion_limit"], int)
    assert config["recursion_limit"] > 0


def test_recursion_limit_is_configurable_via_env(monkeypatch):
    from travel_agent.service import _recursion_limit

    monkeypatch.setenv("TRAVEL_AGENT_MAX_AGENT_STEPS", "7")
    assert _recursion_limit() == 7
    monkeypatch.setenv("TRAVEL_AGENT_MAX_AGENT_STEPS", "not-a-number")
    assert _recursion_limit() == 25
    monkeypatch.setenv("TRAVEL_AGENT_MAX_AGENT_STEPS", "0")
    assert _recursion_limit() == 25


class LoopingModel(BaseChatModel):
    """一直要求调用工具，模拟模型陷入循环。"""

    tool_name: str = "search_hotel_options"
    tool_args: dict[str, Any] = {}
    invocations: int = 0

    @property
    def _llm_type(self):
        return "offline-looping"

    def bind_tools(self, tools, **kwargs):
        return self

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.invocations += 1
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": self.tool_name,
                                "args": self.tool_args,
                                "id": f"call-{self.invocations}",
                                "type": "tool_call",
                            }
                        ],
                    )
                )
            ]
        )


def run_service_chat(monkeypatch, service, consultation_id, model):
    """通过 service 跑一轮 chat，以便验证异常被转成正常结果。"""
    settings = AgentSettings(
        model_mode="openai_compatible", model_name="demo", api_key=None, base_url=None
    )
    monkeypatch.setattr("travel_agent.agent.create_chat_model", lambda _s: model)
    monkeypatch.setattr("travel_agent.service.load_agent_settings", lambda *_: settings)
    return service.chat_consultation(consultation_id, "查询")


def test_budget_stops_a_looping_model(monkeypatch, tmp_path):
    """端到端：模型反复调用工具时，预算必须真正结束本轮。"""
    service, consultation = make_service(tmp_path)
    monkeypatch.setenv("TRAVEL_AGENT_MAX_TOOL_CALLS", "2")
    model = LoopingModel(
        tool_args={
            "city": "杭州",
            "check_in": "2026-10-01",
            "check_out": "2026-10-03",
            "rooms": 1,
        }
    )

    result = run_service_chat(monkeypatch, service, consultation.id, model)

    assert result["status"] == "TOOL_BUDGET_EXCEEDED"
    assert "预算已用尽" in result["answer"]
    # 上限 2：两次执行后第一次拦截返回消息，第二次拦截硬停止
    assert service.repository.count_tool_executions(consultation.id) == 2
    assert "TURN_STOPPED_BY_BUDGET" in event_types(service, consultation.id)


class WrappingUpModel(BaseChatModel):
    """收到错误 ToolMessage 后停止调用工具，模拟配合的模型。"""

    tool_args: dict[str, Any] = {}
    invocations: int = 0
    seen_tool_messages: list[str] = []

    @property
    def _llm_type(self):
        return "offline-wrapping-up"

    def bind_tools(self, tools, **kwargs):
        return self

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.invocations += 1
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        for message in tool_messages:
            self.seen_tool_messages.append(str(message.content))
        if tool_messages:
            return ChatResult(
                generations=[
                    ChatGeneration(message=AIMessage(content="已根据现有信息给出结论"))
                ]
            )
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search_hotel_options",
                                "args": self.tool_args,
                                "id": "call-1",
                                "type": "tool_call",
                            }
                        ],
                    )
                )
            ]
        )


def test_budget_exhaustion_message_reaches_the_model(monkeypatch, tmp_path):
    """预算耗尽必须以错误 ToolMessage 告知模型，而不是静默失败。"""
    service, consultation = make_service(tmp_path)
    # 预先把预算用光，让第一次调用就被拦
    for _ in range(2):
        service.repository.add_audit_event(
            AuditEvent(consultation.id, "TOOL_EXECUTED", {"tool_name": "x"})
        )
    monkeypatch.setenv("TRAVEL_AGENT_MAX_TOOL_CALLS", "2")
    model = WrappingUpModel(
        tool_args={
            "city": "杭州",
            "check_in": "2026-10-01",
            "check_out": "2026-10-03",
            "rooms": 1,
        }
    )

    result = run_service_chat(monkeypatch, service, consultation.id, model)

    assert any("预算已用尽" in text for text in model.seen_tool_messages)
    assert result["status"] == "COMPLETED"


def test_step_limit_is_reported_as_a_normal_result(monkeypatch, tmp_path):
    """步数上限命中时也要给出可读结论，而不是把 GraphRecursionError 抛给顾问。"""
    service, consultation = make_service(tmp_path)
    monkeypatch.setenv("TRAVEL_AGENT_MAX_TOOL_CALLS", "999")
    monkeypatch.setenv("TRAVEL_AGENT_MAX_AGENT_STEPS", "6")
    model = LoopingModel(
        tool_args={
            "city": "杭州",
            "check_in": "2026-10-01",
            "check_out": "2026-10-03",
            "rooms": 1,
        }
    )

    result = run_service_chat(monkeypatch, service, consultation.id, model)

    assert result["status"] == "STEP_LIMIT_EXCEEDED"
    assert "最大推理步数" in result["answer"]
    assert "TURN_STOPPED_BY_STEP_LIMIT" in event_types(service, consultation.id)
    # 停止后咨询必须回到可继续操作的状态
    assert service.get_consultation(consultation.id).status == "OPEN"
