"""仓储接口：应用层依赖它，而不是依赖具体的 SQLite 实现。

## 为什么需要这一层（阶段 9 A2）

``ConsultationService`` 目前把 ``self.repository`` 标注成 ``SQLiteRepository``
这个具体类。这暗示「实现只能是 SQLite」，也让 IDE 补全和类型检查都锚定在
实现细节上。把公开方法抽成 ``Protocol`` 之后，服务层只知道「一个仓储必须能
做什么」——第二阶段要写的 ``PostgresRepository`` 只要结构上匹配就能直接接进来，
服务层一行都不用改。

## 为什么用 ``Protocol`` 而不是抽象基类

``SQLiteRepository`` 已有 1400+ 行。抽象基类要求显式继承，得改类声明；
``Protocol`` 是结构化子类型（PEP 544），**不侵入现有代码**——只要方法签名
对得上就算实现。运行时零开销，只在类型检查期起作用。

## 这一层刻意暴露的耦合

``database_path`` 出现在接口上，而它是 SQLite 特有的概念（Postgres 只有连接串）。
它被保留是因为当前有三处真实依赖：

1. ``checkpoint_path`` 由它派生（``xxx_checkpoints.sqlite``）；
2. ``create_agent_app(...)`` 需要它来装配工具与治理中间件；
3. ``create_travel_tools(...)`` / ``create_sensitive_booking_tools(...)`` 按路径
   各自构造仓储。

**把它如实写在接口上，是为了让耦合可见**——B3 阶段必须先把「工具按路径建仓储」
改成「工具接受仓储或工厂」，否则 Postgres 实现接不进来。假装它不存在只会让
问题推迟到更难调试的地方。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .domain import (
    ApprovalDecision,
    ApprovalRequest,
    AuditEvent,
    Consultation,
    CustomerPreference,
    Message,
    MockBooking,
)


class RepositoryProtocol(Protocol):
    """一个仓储必须提供的全部公开操作。

    方法签名与 ``SQLiteRepository`` 一一对应。新增仓储实现时，这个文件就是
    「需要实现什么」的清单；新增写入方法时，两边都要更新（同时把
    ``tests/test_foreign_keys.py`` 的写入路径列表也补上）。
    """

    #: 数据所在位置。**SQLite 特有**，见模块文档——B3 阶段要消除这个依赖。
    database_path: Path

    # --- 会话 ---------------------------------------------------------------

    def create_consultation(self, consultation: Consultation) -> None: ...

    def get_consultation(self, consultation_id: str) -> Consultation: ...

    def append_message(self, consultation_id: str, message: Message) -> None: ...

    def update_requirements(
        self, consultation_id: str, requirements: dict[str, Any]
    ) -> None: ...

    def update_consultation_status(self, consultation_id: str, status: str) -> None: ...

    # --- 追踪 ---------------------------------------------------------------

    def add_turn_span(
        self,
        *,
        consultation_id: str,
        turn_id: str,
        kind: str,
        name: str,
        duration_ms: int,
        details: dict[str, Any] | None = None,
        sequence: int | None = None,
    ) -> None: ...

    def list_turn_spans(
        self, consultation_id: str, turn_id: str | None = None
    ) -> list[dict[str, Any]]: ...

    # --- 审计与统计 ---------------------------------------------------------

    def add_audit_event(self, event: AuditEvent) -> None: ...

    def list_audit_events(self, consultation_id: str) -> list[AuditEvent]: ...

    def count_tool_executions(self, consultation_id: str) -> int: ...

    def metrics_snapshot(self, *, since: str | None = None) -> dict[str, Any]: ...

    def consult_usage_totals(self, consultation_id: str) -> dict[str, Any]: ...

    # --- 客户偏好 -----------------------------------------------------------

    def set_customer_preference(
        self,
        customer_id: str,
        preference_key: str,
        value: Any,
        *,
        source: str = "explicit",
    ) -> CustomerPreference: ...

    def get_customer_preference(
        self, customer_id: str, preference_key: str
    ) -> CustomerPreference | None: ...

    def list_customer_preferences(self, customer_id: str) -> list[CustomerPreference]: ...

    def delete_customer_preference(
        self, customer_id: str, preference_key: str
    ) -> bool: ...

    # --- 审批 ---------------------------------------------------------------

    def create_approval_request(
        self,
        consultation_id: str,
        thread_id: str,
        tool_name: str,
        tool_arguments: dict[str, Any],
        *,
        requested_by: str,
        risk_level: str = "SENSITIVE",
        expires_at: str | None = None,
        interrupt_id: str | None = None,
        action_index: int | None = None,
    ) -> ApprovalRequest: ...

    def get_approval_request(self, approval_id: str) -> ApprovalRequest: ...

    def list_approval_requests(self, consultation_id: str) -> list[ApprovalRequest]: ...

    def list_pending_approval_requests(
        self, consultation_id: str
    ) -> list[ApprovalRequest]: ...

    def decide_approval_request(
        self,
        approval_id: str,
        decision: ApprovalDecision,
        *,
        decided_by: str,
        reason: str | None = None,
        edited_arguments: dict[str, Any] | None = None,
    ) -> ApprovalRequest: ...

    # --- 预订 ---------------------------------------------------------------

    def create_mock_booking(
        self,
        consultation_id: str,
        itinerary_version_id: str,
        *,
        total_amount: float,
        idempotency_key: str,
    ) -> MockBooking: ...

    def cancel_mock_booking(
        self,
        consultation_id: str,
        booking_id: str,
        *,
        reason: str,
        idempotency_key: str,
    ) -> MockBooking: ...

    def list_mock_bookings(self, consultation_id: str) -> list[MockBooking]: ...

    # --- 方案版本 -----------------------------------------------------------

    def save_itinerary_version(
        self,
        consultation_id: str,
        payload: dict[str, Any],
        tool_history: list[str],
        observations: dict[str, dict[str, Any]],
    ) -> tuple[str, int]: ...

    def list_itinerary_versions(
        self, consultation_id: str
    ) -> list[dict[str, Any]]: ...

    def get_itinerary_version(
        self, consultation_id: str, version_id: str
    ) -> dict[str, Any]: ...

    def select_itinerary_version(
        self, consultation_id: str, version_id: str
    ) -> None: ...

    def get_selected_itinerary(
        self, consultation_id: str
    ) -> dict[str, Any] | None: ...
