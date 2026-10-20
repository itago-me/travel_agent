"""阶段 6.2 E：轮次关联（correlation ID）。

排障的真实痛点：一轮 chat 产生多条审计事件（TOOL_EXECUTED、
ITINERARY_VERSION_RECORDED、AGENT_CHAT_COMPLETED...），原本只靠 created_at
排序。要回答"这一轮到底发生了什么"，只能按时间戳人肉分组。

方案：service 的轮次入口用 ``@correlated`` 绑定一个 turn_id，审计写入时由
``correlation_details()`` 统一合并。事件写入方（工具、中间件、repository）
不需要知道自己在哪一轮。
"""

import pytest

from travel_agent.application import create_consultation_service
from travel_agent.correlation import (
    correlation_details,
    current_turn_id,
    new_turn_id,
    turn_scope,
)
from travel_agent.domain import AuditEvent
from travel_agent.repository import SQLiteRepository


def make(tmp_path, name="t"):
    service = create_consultation_service(tmp_path / f"{name}.db")
    consultation = service.create_consultation("c1", "客户", "去杭州")
    return service, consultation


def events(service, consultation_id):
    return service.repository.list_audit_events(consultation_id)


# --- 关联上下文本身 ---------------------------------------------------------


def test_turn_scope_binds_a_fresh_id_per_entry():
    with turn_scope() as first:
        assert first == current_turn_id()
    with turn_scope() as second:
        assert second == current_turn_id()

    assert first != second


def test_scope_reset_restores_the_outer_context():
    with turn_scope() as outer:
        with turn_scope() as inner:
            assert inner != outer
        assert current_turn_id() == outer, "内层作用域结束后应恢复外层 id"


def test_correlation_details_is_empty_outside_a_turn():
    assert correlation_details() == {}


def test_new_turn_id_binds_until_replaced():
    turn_id = new_turn_id()
    assert current_turn_id() == turn_id
    try:
        assert correlation_details() == {"turn_id": turn_id}
    finally:
        # 不污染其他测试
        from travel_agent.correlation import _turn_id

        _turn_id.set(None)


# --- 审计写入自动携带 --------------------------------------------------------


def test_audits_inside_a_turn_share_one_turn_id(tmp_path):
    service, consultation = make(tmp_path)

    with turn_scope():
        service.repository.add_audit_event(
            AuditEvent(
                consultation_id=consultation.id,
                event_type="TOOL_EXECUTED",
                details={"tool_name": "a"},
            )
        )
        service.repository.add_audit_event(
            AuditEvent(
                consultation_id=consultation.id,
                event_type="TOOL_FAILED",
                details={"tool_name": "b"},
            )
        )

    turn_ids = {
        event.details.get("turn_id")
        for event in events(service, consultation.id)
        if event.event_type in ("TOOL_EXECUTED", "TOOL_FAILED")
    }
    assert turn_ids != {None}, "轮次内写入的事件必须带 turn_id"
    assert len(turn_ids) == 1


def test_audits_outside_a_turn_have_no_turn_id(tmp_path):
    """咨询创建不属于任何轮次，不应带一个 null 字段。"""
    service, consultation = make(tmp_path)

    for event in events(service, consultation.id):
        assert "turn_id" not in event.details


def test_explicit_turn_id_in_details_wins(tmp_path):
    """显式传入的 turn_id 不被上下文覆盖。"""
    repository = SQLiteRepository(tmp_path / "t.db")
    from travel_agent.domain import Consultation

    consultation = Consultation(
        id="c-x", thread_id="th-x", consultant_id="k", customer_id="",
        customer_name="客", status="OPEN", trip_requirements={},
    )
    repository.create_consultation(consultation)

    with turn_scope():
        repository.add_audit_event(
            AuditEvent(
                consultation_id="c-x",
                event_type="TOOL_EXECUTED",
                details={"tool_name": "a", "turn_id": "explicit"},
            )
        )

    stored = repository.list_audit_events("c-x")[0]
    assert stored.details["turn_id"] == "explicit"


# --- 服务入口的轮次归属 ------------------------------------------------------


def test_chat_turn_groups_its_events(monkeypatch, tmp_path):
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from travel_agent.agent import AgentSettings

    class SimpleModel(BaseChatModel):
        invocations: int = 0

        @property
        def _llm_type(self):
            return "simple"

        def bind_tools(self, tools, **kwargs):
            return self

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            return self._generate(messages)

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.invocations += 1
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=f"回答 {self.invocations}"))]
            )

    service, consultation = make(tmp_path)
    monkeypatch.setattr("travel_agent.agent.create_chat_model", lambda _s: SimpleModel())
    monkeypatch.setattr(
        "travel_agent.service.load_agent_settings",
        lambda *_: AgentSettings(
            model_mode="openai_compatible", model_name="demo", api_key=None, base_url=None
        ),
    )

    service.chat_consultation(consultation.id, "第一句")
    service.chat_consultation(consultation.id, "第二句")

    grouped: dict[str | None, list[str]] = {}
    for event in events(service, consultation.id):
        grouped.setdefault(event.details.get("turn_id"), []).append(event.event_type)

    # 创建咨询的两个事件无轮次；两轮 chat 各自一组
    assert grouped[None] == ["CONSULTATION_CREATED", "REQUIREMENTS_EXTRACTED"]
    turn_groups = [v for k, v in grouped.items() if k is not None]
    assert len(turn_groups) == 2
    for group in turn_groups:
        assert "MESSAGE_APPENDED" in group
        assert "AGENT_CHAT_COMPLETED" in group


def test_decide_approval_carries_its_own_turn_id(tmp_path):
    """approve 是独立命令：它的审计事件不应混进 chat 的轮次。"""
    service, consultation = make(tmp_path)
    approval = service.repository.create_approval_request(
        consultation.id,
        consultation.thread_id,
        "create_mock_booking",
        {"consultation_id": consultation.id},
        requested_by="agent",
        interrupt_id="i1",
        action_index=0,
    )

    service.decide_approval(approval.id, "APPROVE", decided_by="c1", resume=False)

    decided = [
        event
        for event in events(service, consultation.id)
        if event.event_type == "APPROVAL_DECIDED"
    ]
    assert decided[0].details.get("turn_id"), "决定事件必须带轮次 id"
