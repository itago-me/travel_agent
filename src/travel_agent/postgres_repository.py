"""PostgreSQL 仓储实现（阶段 9 B3）。

与 ``SQLiteRepository`` 同构，差别集中在四处：

1. **参数占位符**：``?`` → ``%s``（psycopg 用 pyformat）。
2. **连接管理**：SQLite 每次 ``sqlite3.connect()`` 极廉价，Postgres 是网络往返
   （1~5ms），因此用 ``psycopg_pool.ConnectionPool`` 复用；``_connect()`` 是
   上下文管理器，退出时把连接还给池并提交/回滚。
3. **行工厂**：``dict_row`` 让 ``row["col"]`` 的写法与 ``sqlite3.Row`` 一致，
   业务代码不需要改。
4. **没有 schema 创建逻辑**：建表由 Alembic 负责（B1 的 ``0001_initial``）。
   ``SQLiteRepository._initialize`` / ``_enforce_status_constraints`` 在这里
   整体不存在——那是 SQLite 特有的一次性迁移机制。

## 这一版刻意不提供的

``database_path`` 属性（``RepositoryProtocol`` 里声明过）。它是 SQLite 的
「文件路径」概念，Postgres 只有 DSN。但当前有三处真实依赖它：
checkpoint 文件路径派生、``create_agent_app`` 装配、工具按路径自建仓储。

**所以本阶段结束时，``ConsultationService`` 还不能接这个实现**——C1 阶段要先
把「工具与 checkpointer 依赖 SQLite 路径」这件事解开，服务层才能真正切到
Postgres。如实留白，比编一个假的 ``database_path`` 更能反映真实状态。
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator
from uuid import uuid4

from .domain import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    AuditEvent,
    BookingStatus,
    Consultation,
    CustomerPreference,
    Message,
    MockBooking,
    utc_now,
)
from .repository import (
    _amount_tolerance,
    _amount_tolerance_is_strict,
    _percentiles,
)


def ensure_consultation_exists(connection: Any, consultation_id: str) -> None:
    """Raise ``KeyError`` when the consultation row is missing.

    与 SQLite 版的 ``_ensure_exists`` 等价：外键在 Postgres 里是强制的，但把它
    留成应用层错误更友好——调用方拿到的是 ``KeyError`` 而不是
    ``ForeignKeyViolation``，且错误信息里带着具体的 id。
    """
    row = connection.execute(
        "SELECT 1 AS found FROM consultations WHERE id = %s", (consultation_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"Consultation not found: {consultation_id}")


def insert_message(
    connection: Any,
    consultation_id: str,
    message: Message,
    created_at: str | None = None,
) -> None:
    """Insert one message row on an existing connection."""
    connection.execute(
        "INSERT INTO messages (id, consultation_id, role, content, created_at)"
        " VALUES (%s, %s, %s, %s, %s)",
        (
            str(uuid4()),
            consultation_id,
            message.role,
            message.content,
            created_at or utc_now(),
        ),
    )


def plain_dsn(url: str) -> str:
    """把 SQLAlchemy 风格的连接串转成 libpq 认识的形式。

    ``postgresql+psycopg://`` → ``postgresql://``。同一条 DSN 有两种写法：
    Alembic/SQLAlchemy 需要 ``+driver`` 来选方言，而 psycopg（以及
    ``AsyncPostgresSaver``）说的直连 libpq 语法，遇到 ``+`` 会报
    ``missing "=" after ...``（B2 阶段实测踩过）。
    """
    scheme, sep, rest = url.partition("://")
    if not sep:
        return url
    return f"{scheme.split('+')[0]}://{rest}"


class PostgresRepository:
    """仓储的 PostgreSQL 实现。

    构造函数接收 libpq DSN（``postgresql://user:pass@host/db``）——不是
    SQLAlchemy 风格的 ``postgresql+psycopg://``，psycopg 会拒绝后者。
    """

    def __init__(
        self, dsn: str, *, schema: str | None = None, pool_size: int = 4
    ) -> None:
        """``schema`` confines every pooled connection to that schema.

        生产环境用不到（走默认 ``public``）；测试用它对每个用例做隔离——
        连接池里的连接都会带上 ``search_path``，所以仓储不需要在每条 SQL 前面
        写 schema 前缀。
        """
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        self.dsn = dsn
        self.schema = schema
        kwargs: dict[str, Any] = {"row_factory": dict_row}
        if schema:
            # libpq 的 options 参数：-c name=value，注意 -c 后不能有空格
            kwargs["options"] = f"-csearch_path={schema}"
        self._pool = ConnectionPool(
            dsn,
            min_size=1,
            max_size=pool_size,
            kwargs=kwargs,
            open=True,
        )

    def close(self) -> None:
        """关闭连接池。测试与优雅关闭时调用。"""
        self._pool.close()

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        """从池里借一个连接；成功提交，异常回滚，最后归还。

        与 ``sqlite3`` 的 ``with connection:`` 语义对齐：进出边界即事务边界。
        """
        with self._pool.connection() as connection:
            yield connection

    # --- 会话 ---------------------------------------------------------------

    def create_consultation(self, consultation: Consultation) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO consultations
                    (id, thread_id, consultant_id, customer_id, customer_name,
                     status, trip_requirements, planning_constraints,
                     created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    consultation.id,
                    consultation.thread_id,
                    consultation.consultant_id,
                    consultation.customer_id,
                    consultation.customer_name,
                    consultation.status,
                    json.dumps(consultation.trip_requirements, ensure_ascii=False),
                    json.dumps(consultation.planning_constraints, ensure_ascii=False),
                    consultation.created_at,
                    consultation.updated_at,
                ),
            )
            for message in consultation.messages:
                insert_message(connection, consultation.id, message)

    def get_consultation(self, consultation_id: str) -> Consultation:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM consultations WHERE id = %s", (consultation_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Consultation not found: {consultation_id}")
            message_rows = connection.execute(
                "SELECT * FROM messages WHERE consultation_id = %s ORDER BY created_at, id",
                (consultation_id,),
            ).fetchall()
        return Consultation(
            id=row["id"],
            thread_id=row["thread_id"],
            consultant_id=row["consultant_id"],
            customer_id=row["customer_id"],
            customer_name=row["customer_name"],
            status=row["status"],
            trip_requirements=json.loads(row["trip_requirements"]),
            planning_constraints=json.loads(row["planning_constraints"] or "{}"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            messages=[
                Message(role=m["role"], content=m["content"]) for m in message_rows
            ],
        )

    def append_message(self, consultation_id: str, message: Message) -> None:
        now = utc_now()
        with self._connect() as connection:
            ensure_consultation_exists(connection, consultation_id)
            insert_message(connection, consultation_id, message, now)
            connection.execute(
                "UPDATE consultations SET updated_at = %s WHERE id = %s",
                (now, consultation_id),
            )

    def update_requirements(
        self, consultation_id: str, requirements: dict[str, Any]
    ) -> None:
        with self._connect() as connection:
            ensure_consultation_exists(connection, consultation_id)
            connection.execute(
                "UPDATE consultations SET trip_requirements = %s, updated_at = %s"
                " WHERE id = %s",
                (
                    json.dumps(requirements, ensure_ascii=False),
                    utc_now(),
                    consultation_id,
                ),
            )

    def update_consultation_status(self, consultation_id: str, status: str) -> None:
        with self._connect() as connection:
            ensure_consultation_exists(connection, consultation_id)
            connection.execute(
                "UPDATE consultations SET status = %s, updated_at = %s WHERE id = %s",
                (status, utc_now(), consultation_id),
            )

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
    ) -> None:
        with self._connect() as connection:
            ensure_consultation_exists(connection, consultation_id)
            if sequence is None:
                sequence = connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 AS next FROM turn_spans"
                    " WHERE turn_id = %s",
                    (turn_id,),
                ).fetchone()["next"]
            connection.execute(
                """INSERT INTO turn_spans
                   (id, consultation_id, turn_id, sequence, kind, name,
                    duration_ms, details, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    str(uuid4()),
                    consultation_id,
                    turn_id,
                    int(sequence),
                    kind,
                    name,
                    int(duration_ms),
                    json.dumps(details or {}, ensure_ascii=False),
                    utc_now(),
                ),
            )

    def list_turn_spans(
        self, consultation_id: str, turn_id: str | None = None
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM turn_spans WHERE consultation_id = %s"
        params: list[Any] = [consultation_id]
        if turn_id:
            query += " AND turn_id = %s"
            params.append(turn_id)
        # 与 SQLite 实现保持一致的排序理由：并行工具的插入时刻交错，
        # 按 created_at 会打散同一批并发 span。
        query += " ORDER BY turn_id, sequence"
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [
            {
                "turn_id": row["turn_id"],
                "sequence": int(row["sequence"]),
                "kind": row["kind"],
                "name": row["name"],
                "duration_ms": int(row["duration_ms"]),
                "details": json.loads(row["details"] or "{}"),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    # --- 审计与统计 ---------------------------------------------------------

    def add_audit_event(self, event: AuditEvent) -> None:
        with self._connect() as connection:
            ensure_consultation_exists(connection, event.consultation_id)
            self._insert_audit_event(connection, event)

    @staticmethod
    def _insert_audit_event(connection: Any, event: AuditEvent) -> None:
        connection.execute(
            "INSERT INTO audit_events (id, consultation_id, event_type, details, created_at)"
            " VALUES (%s, %s, %s, %s, %s)",
            (
                str(uuid4()),
                event.consultation_id,
                event.event_type,
                json.dumps(event.details, ensure_ascii=False),
                event.created_at,
            ),
        )

    def list_audit_events(self, consultation_id: str) -> list[AuditEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events WHERE consultation_id = %s"
                " ORDER BY created_at, id",
                (consultation_id,),
            ).fetchall()
        return [
            AuditEvent(
                consultation_id=row["consultation_id"],
                event_type=row["event_type"],
                details=json.loads(row["details"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def count_tool_executions(self, consultation_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT COUNT(*) AS total FROM audit_events
                   WHERE consultation_id = %s
                     AND event_type IN ('TOOL_EXECUTED', 'TOOL_FAILED')""",
                (consultation_id,),
            ).fetchone()
        return int(row["total"])

    def metrics_snapshot(self, *, since: str | None = None) -> dict[str, Any]:
        where = "WHERE created_at >= %s" if since else ""
        params: tuple[Any, ...] = (since,) if since else ()
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT event_type, details, created_at FROM audit_events {where}",
                params,
            ).fetchall()

        completed = 0
        stopped: dict[str, int] = {}
        durations: list[int] = []
        input_tokens = output_tokens = 0
        tool_calls = 0
        tool_events = 0

        for row in rows:
            details = json.loads(row["details"]) or {}
            event_type = row["event_type"]
            if event_type == "AGENT_CHAT_COMPLETED":
                completed += 1
                usage = details.get("usage") or {}
                if usage.get("reported"):
                    input_tokens += int(usage.get("input_tokens") or 0)
                    output_tokens += int(usage.get("output_tokens") or 0)
                tool_calls += len(details.get("tool_calls") or [])
            elif event_type.startswith("TURN_STOPPED_BY_"):
                reason = event_type.removeprefix("TURN_STOPPED_BY_").lower()
                stopped[reason] = stopped.get(reason, 0) + 1
            if event_type in ("TOOL_EXECUTED", "TOOL_FAILED"):
                tool_events += 1
            duration = details.get("duration_ms")
            if isinstance(duration, int) and duration >= 0:
                durations.append(duration)

        total_turns = completed + sum(stopped.values())
        return {
            "turns": {
                "total": total_turns,
                "completed": completed,
                "stopped": stopped,
                "failed": sum(stopped.values()),
                "error_rate": (
                    round(sum(stopped.values()) / total_turns, 4) if total_turns else 0.0
                ),
            },
            "latency_ms": _percentiles(durations),
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens,
                "per_turn": (
                    round((input_tokens + output_tokens) / completed) if completed else 0
                ),
            },
            "tools": {
                "calls_total": tool_events,
                "per_turn": round(tool_calls / completed, 2) if completed else 0.0,
            },
            "since": since,
        }

    def consult_usage_totals(self, consultation_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT details FROM audit_events
                   WHERE consultation_id = %s AND event_type = 'AGENT_CHAT_COMPLETED'""",
                (consultation_id,),
            ).fetchall()
        input_tokens = output_tokens = turns = 0
        for row in rows:
            usage = (json.loads(row["details"]) or {}).get("usage") or {}
            if not usage.get("reported"):
                continue
            turns += 1
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
        return {
            "consultation_id": consultation_id,
            "turns": turns,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    # --- 客户偏好 -----------------------------------------------------------

    def set_customer_preference(
        self,
        customer_id: str,
        preference_key: str,
        value: Any,
        *,
        source: str = "explicit",
    ) -> CustomerPreference:
        encoded_value = json.dumps(value, ensure_ascii=False)
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO customer_preferences (
                    customer_id, preference_key, preference_value,
                    source, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT(customer_id, preference_key) DO UPDATE SET
                    preference_value = excluded.preference_value,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (customer_id, preference_key, encoded_value, source, now, now),
            )
            row = connection.execute(
                "SELECT * FROM customer_preferences"
                " WHERE customer_id = %s AND preference_key = %s",
                (customer_id, preference_key),
            ).fetchone()
        return self._customer_preference_from_row(row)

    def get_customer_preference(
        self, customer_id: str, preference_key: str
    ) -> CustomerPreference | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM customer_preferences"
                " WHERE customer_id = %s AND preference_key = %s",
                (customer_id, preference_key),
            ).fetchone()
        return self._customer_preference_from_row(row) if row is not None else None

    def list_customer_preferences(self, customer_id: str) -> list[CustomerPreference]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM customer_preferences"
                " WHERE customer_id = %s ORDER BY preference_key",
                (customer_id,),
            ).fetchall()
        return [self._customer_preference_from_row(row) for row in rows]

    def delete_customer_preference(self, customer_id: str, preference_key: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM customer_preferences"
                " WHERE customer_id = %s AND preference_key = %s",
                (customer_id, preference_key),
            )
            return cursor.rowcount > 0

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
    ) -> ApprovalRequest:
        if not tool_name.strip() or not requested_by.strip():
            raise ValueError("tool_name and requested_by must not be empty")
        if (interrupt_id is None) != (action_index is None):
            raise ValueError("interrupt_id and action_index must be provided together")
        if action_index is not None and action_index < 0:
            raise ValueError("action_index must not be negative")
        approval_id = str(uuid4())
        created_at = utc_now()
        encoded_arguments = json.dumps(
            tool_arguments, ensure_ascii=False, sort_keys=True
        )
        with self._connect() as connection:
            ensure_consultation_exists(connection, consultation_id)
            consultation = connection.execute(
                "SELECT thread_id FROM consultations WHERE id = %s",
                (consultation_id,),
            ).fetchone()
            if consultation["thread_id"] != thread_id:
                raise ValueError("thread_id does not belong to consultation")
            if interrupt_id is not None:
                existing = connection.execute(
                    "SELECT * FROM approval_requests"
                    " WHERE interrupt_id = %s AND action_index = %s",
                    (interrupt_id, action_index),
                ).fetchone()
                if existing is not None:
                    if (
                        existing["consultation_id"] != consultation_id
                        or existing["thread_id"] != thread_id
                        or existing["tool_name"] != tool_name
                        or existing["tool_arguments"] != encoded_arguments
                        or existing["requested_by"] != requested_by
                        or existing["risk_level"] != risk_level
                    ):
                        raise ValueError(
                            "interrupt action reused with different approval payload"
                        )
                    return self._approval_request_from_row(existing)
            connection.execute(
                """
                INSERT INTO approval_requests (
                    id, consultation_id, thread_id, interrupt_id, action_index,
                    tool_name, tool_arguments, risk_level, status, requested_by,
                    created_at, expires_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    approval_id,
                    consultation_id,
                    thread_id,
                    interrupt_id,
                    action_index,
                    tool_name,
                    encoded_arguments,
                    risk_level,
                    ApprovalStatus.PENDING.value,
                    requested_by,
                    created_at,
                    expires_at,
                ),
            )
        return self.get_approval_request(approval_id)

    def get_approval_request(self, approval_id: str) -> ApprovalRequest:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM approval_requests WHERE id = %s", (approval_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Approval request not found: {approval_id}")
        return self._approval_request_from_row(row)

    def list_approval_requests(self, consultation_id: str) -> list[ApprovalRequest]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM approval_requests WHERE consultation_id = %s"
                " ORDER BY created_at, id",
                (consultation_id,),
            ).fetchall()
        return [self._approval_request_from_row(row) for row in rows]

    def list_pending_approval_requests(
        self, consultation_id: str
    ) -> list[ApprovalRequest]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM approval_requests"
                " WHERE consultation_id = %s AND status = %s"
                " ORDER BY created_at, interrupt_id, action_index, id",
                (consultation_id, ApprovalStatus.PENDING.value),
            ).fetchall()
        return [self._approval_request_from_row(row) for row in rows]

    def decide_approval_request(
        self,
        approval_id: str,
        decision: ApprovalDecision,
        *,
        decided_by: str,
        reason: str | None = None,
        edited_arguments: dict[str, Any] | None = None,
    ) -> ApprovalRequest:
        decision = ApprovalDecision(decision)
        decided_by = decided_by.strip()
        reason = reason.strip() if reason else None
        if not decided_by:
            raise ValueError("decided_by must not be empty")
        if decision is ApprovalDecision.REJECT and not reason:
            raise ValueError("rejection reason must not be empty")
        if decision is ApprovalDecision.EDIT and not edited_arguments:
            raise ValueError("edited_arguments must not be empty for EDIT")
        status = (
            ApprovalStatus.APPROVED
            if decision in {ApprovalDecision.APPROVE, ApprovalDecision.EDIT}
            else ApprovalStatus.REJECTED
        )
        now = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE approval_requests
                SET status = %s, decision = %s, decided_by = %s,
                    decision_reason = %s, edited_arguments = %s, decided_at = %s
                WHERE id = %s AND status = %s
                """,
                (
                    status.value,
                    decision.value,
                    decided_by,
                    reason,
                    json.dumps(edited_arguments, ensure_ascii=False, sort_keys=True)
                    if edited_arguments
                    else None,
                    now,
                    approval_id,
                    ApprovalStatus.PENDING.value,
                ),
            )
            if cursor.rowcount == 0:
                row = connection.execute(
                    "SELECT status FROM approval_requests WHERE id = %s",
                    (approval_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"Approval request not found: {approval_id}")
                raise ValueError(f"Approval request already decided: {row['status']}")
        return self.get_approval_request(approval_id)

    # --- 预订 ---------------------------------------------------------------

    def create_mock_booking(
        self,
        consultation_id: str,
        itinerary_version_id: str,
        *,
        total_amount: float,
        idempotency_key: str,
    ) -> MockBooking:
        if total_amount <= 0 or not idempotency_key.strip():
            raise ValueError(
                "total_amount must be positive and idempotency_key required"
            )
        with self._connect() as connection:
            ensure_consultation_exists(connection, consultation_id)
            version = connection.execute(
                """SELECT d.selected_version_id FROM itinerary_versions v
                   JOIN itinerary_drafts d ON d.id = v.draft_id
                   WHERE v.id = %s AND d.consultation_id = %s""",
                (itinerary_version_id, consultation_id),
            ).fetchone()
            if version is None:
                raise KeyError(
                    self._describe_missing_version(
                        connection, consultation_id, itinerary_version_id
                    )
                )
            selected_version_id = version["selected_version_id"]
            if selected_version_id != itinerary_version_id:
                raise ValueError(
                    self._describe_unselected_version(
                        connection,
                        consultation_id,
                        itinerary_version_id,
                        selected_version_id,
                    )
                )
            existing = connection.execute(
                "SELECT * FROM mock_bookings WHERE idempotency_key = %s",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["consultation_id"] != consultation_id
                    or existing["itinerary_version_id"] != itinerary_version_id
                    or float(existing["total_amount"]) != float(total_amount)
                ):
                    raise ValueError(
                        "idempotency_key reused with different booking payload"
                    )
                return self._mock_booking_from_row(existing)

            mismatch = self._check_amount_against_version(
                connection, consultation_id, itinerary_version_id, total_amount
            )
            if mismatch is not None:
                if _amount_tolerance_is_strict() and mismatch["reason"] == (
                    "amount_differs_from_version_budget"
                ):
                    raise ValueError(
                        "booking amount differs from the itinerary version budget: "
                        f"booked {mismatch['booked_total']}, "
                        f"version {mismatch['version_total']}, "
                        f"delta {mismatch['delta']}. "
                        "Either book the quoted amount or unset "
                        "TRAVEL_AGENT_BOOKING_AMOUNT_STRICT to record the difference."
                    )
                self._insert_audit_event(
                    connection,
                    AuditEvent(
                        consultation_id=consultation_id,
                        event_type="APPROVAL_AMOUNT_MISMATCH",
                        details={
                            "itinerary_version_id": itinerary_version_id,
                            **mismatch,
                            "idempotency_key": idempotency_key,
                        },
                    ),
                )
            now = utc_now()
            booking_id = str(uuid4())
            connection.execute(
                """INSERT INTO mock_bookings (
                    id, consultation_id, itinerary_version_id, total_amount,
                    status, idempotency_key, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    booking_id,
                    consultation_id,
                    itinerary_version_id,
                    total_amount,
                    BookingStatus.CREATED.value,
                    idempotency_key,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM mock_bookings WHERE id = %s", (booking_id,)
            ).fetchone()
            self._insert_audit_event(
                connection,
                AuditEvent(
                    consultation_id=consultation_id,
                    event_type="BOOKING_CREATED",
                    details={
                        "booking_id": booking_id,
                        "itinerary_version_id": itinerary_version_id,
                        "total_amount": float(total_amount),
                        "idempotency_key": idempotency_key,
                    },
                ),
            )
        return self._mock_booking_from_row(row)

    def cancel_mock_booking(
        self,
        consultation_id: str,
        booking_id: str,
        *,
        reason: str,
        idempotency_key: str,
    ) -> MockBooking:
        reason = reason.strip()
        if not reason or not idempotency_key.strip():
            raise ValueError("reason and idempotency_key must not be empty")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM mock_bookings WHERE id = %s AND consultation_id = %s",
                (booking_id, consultation_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"Mock booking not found: {booking_id}")
            if row["status"] == BookingStatus.CANCELLED.value:
                if row["cancellation_idempotency_key"] != idempotency_key:
                    raise ValueError("booking already cancelled by another request")
                return self._mock_booking_from_row(row)
            now = utc_now()
            connection.execute(
                """UPDATE mock_bookings SET status = %s, cancellation_idempotency_key = %s,
                   cancellation_reason = %s, cancelled_at = %s, updated_at = %s
                   WHERE id = %s""",
                (
                    BookingStatus.CANCELLED.value,
                    idempotency_key,
                    reason,
                    now,
                    now,
                    booking_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM mock_bookings WHERE id = %s", (booking_id,)
            ).fetchone()
            self._insert_audit_event(
                connection,
                AuditEvent(
                    consultation_id=consultation_id,
                    event_type="BOOKING_CANCELLED",
                    details={
                        "booking_id": booking_id,
                        "cancellation_reason": reason,
                        "idempotency_key": idempotency_key,
                    },
                ),
            )
        return self._mock_booking_from_row(row)

    def list_mock_bookings(self, consultation_id: str) -> list[MockBooking]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM mock_bookings WHERE consultation_id = %s"
                " ORDER BY created_at, id",
                (consultation_id,),
            ).fetchall()
        return [self._mock_booking_from_row(row) for row in rows]

    # --- 方案版本 -----------------------------------------------------------

    def save_itinerary_version(
        self,
        consultation_id: str,
        payload: dict[str, Any],
        tool_history: list[str],
        observations: dict[str, dict[str, Any]],
    ) -> tuple[str, int]:
        now = utc_now()
        with self._connect() as connection:
            ensure_consultation_exists(connection, consultation_id)
            draft = connection.execute(
                "SELECT id FROM itinerary_drafts WHERE consultation_id = %s",
                (consultation_id,),
            ).fetchone()
            if draft is None:
                draft_id = str(uuid4())
                connection.execute(
                    "INSERT INTO itinerary_drafts"
                    " (id, consultation_id, created_at, updated_at)"
                    " VALUES (%s, %s, %s, %s)",
                    (draft_id, consultation_id, now, now),
                )
            else:
                draft_id = draft["id"]
                connection.execute(
                    "UPDATE itinerary_drafts SET updated_at = %s WHERE id = %s",
                    (now, draft_id),
                )
            row = connection.execute(
                "SELECT COALESCE(MAX(version_number), 0) AS latest FROM itinerary_versions"
                " WHERE draft_id = %s",
                (draft_id,),
            ).fetchone()
            version_number = int(row["latest"]) + 1
            version_id = str(uuid4())
            connection.execute(
                "INSERT INTO itinerary_versions"
                " (id, draft_id, version_number, payload, created_at)"
                " VALUES (%s, %s, %s, %s, %s)",
                (
                    version_id,
                    draft_id,
                    version_number,
                    json.dumps(payload, ensure_ascii=False),
                    now,
                ),
            )
            for call_number, tool_name in enumerate(tool_history, start=1):
                connection.execute(
                    "INSERT INTO provider_requests"
                    " (id, consultation_id, version_id, tool_name, call_number,"
                    "  response_payload, created_at)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        str(uuid4()),
                        consultation_id,
                        version_id,
                        tool_name,
                        call_number,
                        json.dumps(observations.get(tool_name, {}), ensure_ascii=False),
                        now,
                    ),
                )
        return version_id, version_number

    def list_itinerary_versions(self, consultation_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT v.id, v.version_number, v.payload, v.created_at
                   FROM itinerary_versions v
                   JOIN itinerary_drafts d ON d.id = v.draft_id
                   WHERE d.consultation_id = %s ORDER BY v.version_number""",
                (consultation_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "version_number": row["version_number"],
                "payload": json.loads(row["payload"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_itinerary_version(
        self, consultation_id: str, version_id: str
    ) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT v.id, v.version_number, v.payload, v.created_at
                   FROM itinerary_versions v
                   JOIN itinerary_drafts d ON d.id = v.draft_id
                   WHERE d.consultation_id = %s AND v.id = %s""",
                (consultation_id, version_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"Itinerary version not found: {version_id}")
        return {
            "id": row["id"],
            "version_number": row["version_number"],
            "payload": json.loads(row["payload"]),
            "created_at": row["created_at"],
        }

    def select_itinerary_version(self, consultation_id: str, version_id: str) -> None:
        now = utc_now()
        with self._connect() as connection:
            draft = connection.execute(
                "SELECT id FROM itinerary_drafts WHERE consultation_id = %s",
                (consultation_id,),
            ).fetchone()
            if draft is None:
                raise KeyError(f"Itinerary draft not found: {consultation_id}")
            version = connection.execute(
                "SELECT 1 AS found FROM itinerary_versions WHERE id = %s AND draft_id = %s",
                (version_id, draft["id"]),
            ).fetchone()
            if version is None:
                raise KeyError(f"Itinerary version not found: {version_id}")
            connection.execute(
                "UPDATE itinerary_drafts SET selected_version_id = %s,"
                " selected_at = %s, updated_at = %s WHERE id = %s",
                (version_id, now, now, draft["id"]),
            )

    def get_selected_itinerary(self, consultation_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT selected_version_id, selected_at FROM itinerary_drafts"
                " WHERE consultation_id = %s",
                (consultation_id,),
            ).fetchone()
        if row is None or row["selected_version_id"] is None:
            return None
        selected = self.get_itinerary_version(
            consultation_id, row["selected_version_id"]
        )
        selected["selected_at"] = row["selected_at"]
        selected["version_id"] = selected.pop("id")
        return selected

    # --- 内部辅助 -----------------------------------------------------------

    def _version_summaries(self, connection: Any, consultation_id: str) -> str:
        rows = connection.execute(
            """SELECT v.id, v.version_number
               FROM itinerary_versions v
               JOIN itinerary_drafts d ON d.id = v.draft_id
               WHERE d.consultation_id = %s
               ORDER BY v.version_number""",
            (consultation_id,),
        ).fetchall()
        return (
            ", ".join(f"v{row['version_number']}={row['id']}" for row in rows) or "none"
        )

    def _describe_missing_version(
        self,
        connection: Any,
        consultation_id: str,
        itinerary_version_id: str,
    ) -> str:
        available = self._version_summaries(connection, consultation_id)
        if available == "none":
            return (
                f"itinerary version {itinerary_version_id} not found in consultation "
                f"{consultation_id}, and that consultation has no itinerary version yet. "
                "Run a chat turn that searches transport and hotels first."
            )
        return (
            f"itinerary version {itinerary_version_id} not found in consultation "
            f"{consultation_id}. Available versions: {available}."
        )

    def _describe_unselected_version(
        self,
        connection: Any,
        consultation_id: str,
        itinerary_version_id: str,
        selected_version_id: str | None,
    ) -> str:
        available = self._version_summaries(connection, consultation_id)
        selected = (
            f"selected version is {selected_version_id}"
            if selected_version_id
            else "no version is selected yet"
        )
        fix = f"travel-agent select-itinerary {consultation_id} {itinerary_version_id}"
        return (
            f"itinerary version must be selected before booking: "
            f"requested {itinerary_version_id}, but {selected}. "
            f"Available versions: {available}. "
            f"Run `{fix}` and retry the booking."
        )

    def _check_amount_against_version(
        self,
        connection: Any,
        consultation_id: str,
        itinerary_version_id: str,
        total_amount: float,
    ) -> dict[str, Any] | None:
        row = connection.execute(
            "SELECT payload FROM itinerary_versions WHERE id = %s",
            (itinerary_version_id,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload"])
        frozen_total = (payload.get("budget") or {}).get("total")
        if frozen_total is None:
            return {
                "reason": "version_budget_missing",
                "booked_total": float(total_amount),
                "version_total": None,
                "delta": None,
            }
        frozen_total = float(frozen_total)
        delta = float(total_amount) - frozen_total
        if frozen_total == 0:
            exceeded = abs(delta) > 0
        else:
            exceeded = abs(delta) / abs(frozen_total) > _amount_tolerance()
        if not exceeded:
            return None
        return {
            "reason": "amount_differs_from_version_budget",
            "booked_total": float(total_amount),
            "version_total": frozen_total,
            "delta": delta,
            "tolerance_ratio": _amount_tolerance(),
        }

    @staticmethod
    def _customer_preference_from_row(row: dict[str, Any]) -> CustomerPreference:
        return CustomerPreference(
            customer_id=row["customer_id"],
            preference_key=row["preference_key"],
            value=json.loads(row["preference_value"]),
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _approval_request_from_row(row: dict[str, Any]) -> ApprovalRequest:
        return ApprovalRequest(
            id=row["id"],
            consultation_id=row["consultation_id"],
            thread_id=row["thread_id"],
            interrupt_id=row["interrupt_id"],
            action_index=row["action_index"],
            tool_name=row["tool_name"],
            tool_arguments=json.loads(row["tool_arguments"]),
            risk_level=row["risk_level"],
            status=ApprovalStatus(row["status"]),
            requested_by=row["requested_by"],
            created_at=row["created_at"],
            decision=ApprovalDecision(row["decision"]) if row["decision"] else None,
            decided_by=row["decided_by"],
            decision_reason=row["decision_reason"],
            edited_arguments=(
                json.loads(row["edited_arguments"]) if row["edited_arguments"] else None
            ),
            decided_at=row["decided_at"],
            expires_at=row["expires_at"],
        )

    @staticmethod
    def _mock_booking_from_row(row: dict[str, Any]) -> MockBooking:
        return MockBooking(
            id=row["id"],
            consultation_id=row["consultation_id"],
            itinerary_version_id=row["itinerary_version_id"],
            total_amount=float(row["total_amount"]),
            status=BookingStatus(row["status"]),
            idempotency_key=row["idempotency_key"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            cancellation_idempotency_key=row["cancellation_idempotency_key"],
            cancellation_reason=row["cancellation_reason"],
            cancelled_at=row["cancelled_at"],
        )
