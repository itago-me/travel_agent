import json
import sqlite3
from pathlib import Path
from typing import Any
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


class SQLiteRepository:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS consultations (
                    id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL UNIQUE,
                    consultant_id TEXT NOT NULL,
                    customer_id TEXT NOT NULL DEFAULT '',
                    customer_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    trip_requirements TEXT NOT NULL,
                    planning_constraints TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    consultation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (consultation_id) REFERENCES consultations(id)
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    consultation_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    details TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (consultation_id) REFERENCES consultations(id)
                );
                CREATE TABLE IF NOT EXISTS itinerary_drafts (
                    id TEXT PRIMARY KEY,
                    consultation_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    selected_version_id TEXT,
                    selected_at TEXT,
                    FOREIGN KEY (consultation_id) REFERENCES consultations(id)
                );
                CREATE TABLE IF NOT EXISTS itinerary_versions (
                    id TEXT PRIMARY KEY,
                    draft_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (draft_id, version_number),
                    FOREIGN KEY (draft_id) REFERENCES itinerary_drafts(id)
                );
                CREATE TABLE IF NOT EXISTS provider_requests (
                    id TEXT PRIMARY KEY,
                    consultation_id TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    call_number INTEGER NOT NULL,
                    response_payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (consultation_id) REFERENCES consultations(id),
                    FOREIGN KEY (version_id) REFERENCES itinerary_versions(id)
                );
                CREATE TABLE IF NOT EXISTS customer_preferences (
                    customer_id TEXT NOT NULL,
                    preference_key TEXT NOT NULL,
                    preference_value TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (customer_id, preference_key)
                );
                CREATE TABLE IF NOT EXISTS approval_requests (
                    id TEXT PRIMARY KEY,
                    consultation_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    interrupt_id TEXT,
                    action_index INTEGER,
                    tool_name TEXT NOT NULL,
                    tool_arguments TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK (status IN ('PENDING','APPROVED','REJECTED','EXPIRED','CANCELLED')),
                    requested_by TEXT NOT NULL,
                    decision TEXT
                        CHECK (decision IS NULL OR decision IN ('APPROVE','EDIT','REJECT')),
                    decided_by TEXT,
                    decision_reason TEXT,
                    edited_arguments TEXT,
                    created_at TEXT NOT NULL,
                    decided_at TEXT,
                    expires_at TEXT,
                    FOREIGN KEY (consultation_id) REFERENCES consultations(id)
                );
                CREATE TABLE IF NOT EXISTS mock_bookings (
                    id TEXT PRIMARY KEY,
                    consultation_id TEXT NOT NULL,
                    itinerary_version_id TEXT NOT NULL,
                    total_amount REAL NOT NULL,
                    status TEXT NOT NULL
                        CHECK (status IN ('CREATED','CANCELLED','FAILED')),
                    idempotency_key TEXT NOT NULL UNIQUE,
                    cancellation_idempotency_key TEXT UNIQUE,
                    cancellation_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    cancelled_at TEXT,
                    FOREIGN KEY (consultation_id) REFERENCES consultations(id),
                    FOREIGN KEY (itinerary_version_id) REFERENCES itinerary_versions(id)
                );
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(consultations)")
            }
            if "planning_constraints" not in columns:
                connection.execute(
                    "ALTER TABLE consultations ADD COLUMN planning_constraints TEXT NOT NULL DEFAULT '{}'"
                )
            if "customer_id" not in columns:
                connection.execute(
                    "ALTER TABLE consultations ADD COLUMN customer_id TEXT NOT NULL DEFAULT ''"
                )
            draft_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(itinerary_drafts)")
            }
            if "selected_version_id" not in draft_columns:
                connection.execute(
                    "ALTER TABLE itinerary_drafts ADD COLUMN selected_version_id TEXT"
                )
            if "selected_at" not in draft_columns:
                connection.execute(
                    "ALTER TABLE itinerary_drafts ADD COLUMN selected_at TEXT"
                )
            approval_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(approval_requests)")
            }
            if "edited_arguments" not in approval_columns:
                connection.execute(
                    "ALTER TABLE approval_requests ADD COLUMN edited_arguments TEXT"
                )
            if "interrupt_id" not in approval_columns:
                connection.execute(
                    "ALTER TABLE approval_requests ADD COLUMN interrupt_id TEXT"
                )
            if "action_index" not in approval_columns:
                connection.execute(
                    "ALTER TABLE approval_requests ADD COLUMN action_index INTEGER"
                )
            connection.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS
                   approval_requests_interrupt_action
                   ON approval_requests(interrupt_id, action_index)
                   WHERE interrupt_id IS NOT NULL"""
            )
            self._enforce_status_constraints(connection)

    # 旧库里的 status / decision 没有取值约束，脏数据会让 ApprovalStatus(...)
    # 在读取时抛枚举异常，整条审批链路卡死。SQLite 不支持 ALTER TABLE ADD
    # CONSTRAINT，只能按官方推荐重建表。
    _CONSTRAINED_TABLES = {
        "approval_requests": (
            """CREATE TABLE approval_requests__new (
                id TEXT PRIMARY KEY,
                consultation_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                interrupt_id TEXT,
                action_index INTEGER,
                tool_name TEXT NOT NULL,
                tool_arguments TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                status TEXT NOT NULL
                    CHECK (status IN ('PENDING','APPROVED','REJECTED','EXPIRED','CANCELLED')),
                requested_by TEXT NOT NULL,
                decision TEXT
                    CHECK (decision IS NULL OR decision IN ('APPROVE','EDIT','REJECT')),
                decided_by TEXT,
                decision_reason TEXT,
                edited_arguments TEXT,
                created_at TEXT NOT NULL,
                decided_at TEXT,
                expires_at TEXT,
                FOREIGN KEY (consultation_id) REFERENCES consultations(id)
            )""",
            ("id", "consultation_id", "thread_id", "interrupt_id", "action_index",
             "tool_name", "tool_arguments", "risk_level", "status", "requested_by",
             "decision", "decided_by", "decision_reason", "edited_arguments",
             "created_at", "decided_at", "expires_at"),
        ),
        "mock_bookings": (
            """CREATE TABLE mock_bookings__new (
                id TEXT PRIMARY KEY,
                consultation_id TEXT NOT NULL,
                itinerary_version_id TEXT NOT NULL,
                total_amount REAL NOT NULL,
                status TEXT NOT NULL
                    CHECK (status IN ('CREATED','CANCELLED','FAILED')),
                idempotency_key TEXT NOT NULL UNIQUE,
                cancellation_idempotency_key TEXT UNIQUE,
                cancellation_reason TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                cancelled_at TEXT,
                FOREIGN KEY (consultation_id) REFERENCES consultations(id),
                FOREIGN KEY (itinerary_version_id) REFERENCES itinerary_versions(id)
            )""",
            ("id", "consultation_id", "itinerary_version_id", "total_amount",
             "status", "idempotency_key", "cancellation_idempotency_key",
             "cancellation_reason", "created_at", "updated_at", "cancelled_at"),
        ),
    }

    def _enforce_status_constraints(self, connection: sqlite3.Connection) -> None:
        """Rebuild tables that predate the CHECK constraints, preserving rows."""
        # 重建期间必须关闭外键，否则 DROP 旧表会触发引用检查。
        connection.execute("PRAGMA foreign_keys = OFF")
        try:
            for table, (create_sql, columns) in self._CONSTRAINED_TABLES.items():
                existing = connection.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                if existing is None:
                    continue
                if "CHECK" in (existing["sql"] or ""):
                    continue  # 已是新结构
                temp = f"{table}__new"
                connection.execute(f"DROP TABLE IF EXISTS {temp}")
                connection.execute(create_sql)
                column_list = ", ".join(columns)
                # 非法值统一归一到 PENDING：这些行从未被决定过，而卡在非法
                # 状态上会让顾问永远无法处理它们。真正的历史结论（APPROVED
                # 等）保留原值，只修正无法识别的写坏值。
                if table == "approval_requests":
                    connection.execute(
                        f"""INSERT INTO {temp} ({column_list})
                            SELECT
                                id, consultation_id, thread_id, interrupt_id,
                                action_index, tool_name, tool_arguments, risk_level,
                                CASE
                                    WHEN status IN ('PENDING','APPROVED','REJECTED','EXPIRED','CANCELLED')
                                        THEN status
                                    ELSE 'PENDING'
                                END,
                                requested_by,
                                CASE
                                    WHEN decision IN ('APPROVE','EDIT','REJECT') THEN decision
                                    ELSE NULL
                                END,
                                decided_by, decision_reason, edited_arguments,
                                created_at, decided_at, expires_at
                            FROM {table}"""
                    )
                else:
                    connection.execute(
                        f"""INSERT INTO {temp} ({column_list})
                            SELECT
                                id, consultation_id, itinerary_version_id,
                                total_amount,
                                CASE
                                    WHEN status IN ('CREATED','CANCELLED','FAILED') THEN status
                                    ELSE 'FAILED'
                                END,
                                idempotency_key, cancellation_idempotency_key,
                                cancellation_reason, created_at, updated_at, cancelled_at
                            FROM {table}"""
                    )
                connection.execute(f"DROP TABLE {table}")
                connection.execute(f"ALTER TABLE {temp} RENAME TO {table}")
        finally:
            connection.execute("PRAGMA foreign_keys = ON")
        # 重建会丢掉旧索引，这里补回（建表语句只带 UNIQUE 内联约束）。
        connection.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS
               approval_requests_interrupt_action
               ON approval_requests(interrupt_id, action_index)
               WHERE interrupt_id IS NOT NULL"""
        )

    def create_consultation(self, consultation: Consultation) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO consultations
                    (id, thread_id, consultant_id, customer_id, customer_name,
                     status, trip_requirements, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    consultation.id,
                    consultation.thread_id,
                    consultation.consultant_id,
                    consultation.customer_id,
                    consultation.customer_name,
                    consultation.status,
                    json.dumps(consultation.trip_requirements, ensure_ascii=False),
                    consultation.created_at,
                    consultation.updated_at,
                ),
            )
            for message in consultation.messages:
                self._insert_message(connection, consultation.id, message)

    def get_consultation(self, consultation_id: str) -> Consultation:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM consultations WHERE id = ?", (consultation_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Consultation not found: {consultation_id}")
            messages = connection.execute(
                "SELECT role, content FROM messages WHERE consultation_id = ? ORDER BY created_at, id",
                (consultation_id,),
            ).fetchall()
        return Consultation(
            id=row["id"],
            thread_id=row["thread_id"],
            consultant_id=row["consultant_id"],
            customer_name=row["customer_name"],
            status=row["status"],
            customer_id=row["customer_id"] or f"legacy:{row['customer_name']}",
            messages=[Message(item["role"], item["content"]) for item in messages],
            trip_requirements=json.loads(row["trip_requirements"]),
            planning_constraints=json.loads(row["planning_constraints"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def append_message(self, consultation_id: str, message: Message) -> None:
        now = utc_now()
        with self._connect() as connection:
            self._ensure_exists(connection, consultation_id)
            self._insert_message(connection, consultation_id, message, now)
            connection.execute(
                "UPDATE consultations SET updated_at = ? WHERE id = ?",
                (now, consultation_id),
            )

    def update_requirements(
        self, consultation_id: str, requirements: dict[str, Any]
    ) -> None:
        with self._connect() as connection:
            self._ensure_exists(connection, consultation_id)
            connection.execute(
                "UPDATE consultations SET trip_requirements = ?, updated_at = ? WHERE id = ?",
                (
                    json.dumps(requirements, ensure_ascii=False),
                    utc_now(),
                    consultation_id,
                ),
            )

    def update_consultation_status(self, consultation_id: str, status: str) -> None:
        status = status.strip()
        if not status:
            raise ValueError("status must not be empty")
        with self._connect() as connection:
            self._ensure_exists(connection, consultation_id)
            connection.execute(
                "UPDATE consultations SET status = ?, updated_at = ? WHERE id = ?",
                (status, utc_now(), consultation_id),
            )

    def add_audit_event(self, event: AuditEvent) -> None:
        with self._connect() as connection:
            self._ensure_exists(connection, event.consultation_id)
            connection.execute(
                "INSERT INTO audit_events (id, consultation_id, event_type, details, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    event.consultation_id,
                    event.event_type,
                    json.dumps(event.details, ensure_ascii=False),
                    event.created_at,
                ),
            )

    def set_customer_preference(
        self,
        customer_id: str,
        preference_key: str,
        value: Any,
        *,
        source: str = "explicit",
    ) -> CustomerPreference:
        customer_id = customer_id.strip()
        preference_key = preference_key.strip()
        source = source.strip()
        if not customer_id:
            raise ValueError("customer_id must not be empty")
        if not preference_key:
            raise ValueError("preference_key must not be empty")
        if not source:
            raise ValueError("source must not be empty")
        encoded_value = json.dumps(value, ensure_ascii=False)
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO customer_preferences (
                    customer_id, preference_key, preference_value,
                    source, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(customer_id, preference_key) DO UPDATE SET
                    preference_value = excluded.preference_value,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (
                    customer_id,
                    preference_key,
                    encoded_value,
                    source,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM customer_preferences
                WHERE customer_id = ? AND preference_key = ?
                """,
                (customer_id, preference_key),
            ).fetchone()
        return self._customer_preference_from_row(row)

    def get_customer_preference(
        self, customer_id: str, preference_key: str
    ) -> CustomerPreference | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM customer_preferences
                WHERE customer_id = ? AND preference_key = ?
                """,
                (customer_id, preference_key),
            ).fetchone()
        return self._customer_preference_from_row(row) if row is not None else None

    def list_customer_preferences(self, customer_id: str) -> list[CustomerPreference]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM customer_preferences
                WHERE customer_id = ? ORDER BY preference_key
                """,
                (customer_id,),
            ).fetchall()
        return [self._customer_preference_from_row(row) for row in rows]

    def delete_customer_preference(self, customer_id: str, preference_key: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM customer_preferences
                WHERE customer_id = ? AND preference_key = ?
                """,
                (customer_id, preference_key),
            )
        return cursor.rowcount > 0

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
            self._ensure_exists(connection, consultation_id)
            consultation = connection.execute(
                "SELECT thread_id FROM consultations WHERE id = ?", (consultation_id,)
            ).fetchone()
            if consultation["thread_id"] != thread_id:
                raise ValueError("thread_id does not belong to consultation")
            if interrupt_id is not None:
                existing = connection.execute(
                    """SELECT * FROM approval_requests
                       WHERE interrupt_id = ? AND action_index = ?""",
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
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                "SELECT * FROM approval_requests WHERE id = ?", (approval_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Approval request not found: {approval_id}")
        return self._approval_request_from_row(row)

    def list_approval_requests(self, consultation_id: str) -> list[ApprovalRequest]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM approval_requests
                   WHERE consultation_id = ? ORDER BY created_at, id""",
                (consultation_id,),
            ).fetchall()
        return [self._approval_request_from_row(row) for row in rows]

    def list_pending_approval_requests(
        self, consultation_id: str
    ) -> list[ApprovalRequest]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM approval_requests
                   WHERE consultation_id = ? AND status = ?
                   ORDER BY created_at, interrupt_id, action_index, id""",
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
                SET status = ?, decision = ?, decided_by = ?,
                    decision_reason = ?, edited_arguments = ?, decided_at = ?
                WHERE id = ? AND status = ?
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
                    "SELECT status FROM approval_requests WHERE id = ?", (approval_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"Approval request not found: {approval_id}")
                raise ValueError(f"Approval request already decided: {row['status']}")
        return self.get_approval_request(approval_id)

    def _describe_missing_version(
        self,
        connection: sqlite3.Connection,
        consultation_id: str,
        itinerary_version_id: str,
    ) -> str:
        """Explain an unknown version id, listing what is actually available."""
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
        connection: sqlite3.Connection,
        consultation_id: str,
        itinerary_version_id: str,
        selected_version_id: str | None,
    ) -> str:
        """Explain the selection mismatch and print the exact command that fixes it."""
        available = self._version_summaries(connection, consultation_id)
        selected = (
            f"selected version is {selected_version_id}"
            if selected_version_id
            else "no version is selected yet"
        )
        fix = (
            f"travel-agent select-itinerary {consultation_id} {itinerary_version_id}"
        )
        return (
            f"itinerary version must be selected before booking: "
            f"requested {itinerary_version_id}, but {selected}. "
            f"Available versions: {available}. "
            f"Run `{fix}` and retry the booking."
        )

    def _version_summaries(
        self, connection: sqlite3.Connection, consultation_id: str
    ) -> str:
        """Compact 'v1=id' listing used inside error messages."""
        rows = connection.execute(
            """SELECT v.id, v.version_number
               FROM itinerary_versions v
               JOIN itinerary_drafts d ON d.id = v.draft_id
               WHERE d.consultation_id = ?
               ORDER BY v.version_number""",
            (consultation_id,),
        ).fetchall()
        return ", ".join(
            f"v{row['version_number']}={row['id']}" for row in rows
        ) or "none"

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
            self._ensure_exists(connection, consultation_id)
            version = connection.execute(
                """SELECT d.selected_version_id FROM itinerary_versions v
                   JOIN itinerary_drafts d ON d.id = v.draft_id
                   WHERE v.id = ? AND d.consultation_id = ?""",
                (itinerary_version_id, consultation_id),
            ).fetchone()
            if version is None:
                raise KeyError(
                    self._describe_missing_version(connection, consultation_id, itinerary_version_id)
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
                "SELECT * FROM mock_bookings WHERE idempotency_key = ?",
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
            now = utc_now()
            booking_id = str(uuid4())
            connection.execute(
                """INSERT INTO mock_bookings (
                    id, consultation_id, itinerary_version_id, total_amount,
                    status, idempotency_key, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
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
                "SELECT * FROM mock_bookings WHERE id = ?", (booking_id,)
            ).fetchone()
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
                """SELECT * FROM mock_bookings
                   WHERE id = ? AND consultation_id = ?""",
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
                """UPDATE mock_bookings SET status = ?, cancellation_idempotency_key = ?,
                   cancellation_reason = ?, cancelled_at = ?, updated_at = ?
                   WHERE id = ?""",
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
                "SELECT * FROM mock_bookings WHERE id = ?", (booking_id,)
            ).fetchone()
        return self._mock_booking_from_row(row)

    def list_mock_bookings(self, consultation_id: str) -> list[MockBooking]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM mock_bookings
                   WHERE consultation_id = ? ORDER BY created_at, id""",
                (consultation_id,),
            ).fetchall()
        return [self._mock_booking_from_row(row) for row in rows]

    def list_audit_events(self, consultation_id: str) -> list[AuditEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events WHERE consultation_id = ? ORDER BY created_at, id",
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

    @staticmethod
    def _customer_preference_from_row(row: sqlite3.Row) -> CustomerPreference:
        return CustomerPreference(
            customer_id=row["customer_id"],
            preference_key=row["preference_key"],
            value=json.loads(row["preference_value"]),
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _approval_request_from_row(row: sqlite3.Row) -> ApprovalRequest:
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
    def _mock_booking_from_row(row: sqlite3.Row) -> MockBooking:
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

    def save_itinerary_version(
        self,
        consultation_id: str,
        payload: dict[str, Any],
        tool_history: list[str],
        observations: dict[str, dict[str, Any]],
    ) -> tuple[str, int]:
        now = utc_now()
        with self._connect() as connection:
            self._ensure_exists(connection, consultation_id)
            draft = connection.execute(
                "SELECT id FROM itinerary_drafts WHERE consultation_id = ?",
                (consultation_id,),
            ).fetchone()
            if draft is None:
                draft_id = str(uuid4())
                connection.execute(
                    "INSERT INTO itinerary_drafts (id, consultation_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (draft_id, consultation_id, now, now),
                )
            else:
                draft_id = draft["id"]
                connection.execute(
                    "UPDATE itinerary_drafts SET updated_at = ? WHERE id = ?",
                    (now, draft_id),
                )
            row = connection.execute(
                "SELECT COALESCE(MAX(version_number), 0) AS latest FROM itinerary_versions WHERE draft_id = ?",
                (draft_id,),
            ).fetchone()
            version_number = int(row["latest"]) + 1
            version_id = str(uuid4())
            connection.execute(
                "INSERT INTO itinerary_versions (id, draft_id, version_number, payload, created_at) VALUES (?, ?, ?, ?, ?)",
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
                    "INSERT INTO provider_requests (id, consultation_id, version_id, tool_name, call_number, response_payload, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
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
                   FROM itinerary_versions v JOIN itinerary_drafts d ON d.id = v.draft_id
                   WHERE d.consultation_id = ? ORDER BY v.version_number""",
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
                   FROM itinerary_versions v JOIN itinerary_drafts d ON d.id = v.draft_id
                   WHERE d.consultation_id = ? AND v.id = ?""",
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
                "SELECT id FROM itinerary_drafts WHERE consultation_id = ?",
                (consultation_id,),
            ).fetchone()
            if draft is None:
                raise KeyError(f"Itinerary draft not found: {consultation_id}")
            version = connection.execute(
                "SELECT 1 FROM itinerary_versions WHERE id = ? AND draft_id = ?",
                (version_id, draft["id"]),
            ).fetchone()
            if version is None:
                raise KeyError(f"Itinerary version not found: {version_id}")
            connection.execute(
                "UPDATE itinerary_drafts SET selected_version_id = ?, selected_at = ?, updated_at = ? WHERE id = ?",
                (version_id, now, now, draft["id"]),
            )

    def get_selected_itinerary(self, consultation_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT selected_version_id, selected_at FROM itinerary_drafts WHERE consultation_id = ?",
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

    @staticmethod
    def _insert_message(
        connection: sqlite3.Connection,
        consultation_id: str,
        message: Message,
        created_at: str | None = None,
    ) -> None:
        connection.execute(
            "INSERT INTO messages (id, consultation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                str(uuid4()),
                consultation_id,
                message.role,
                message.content,
                created_at or utc_now(),
            ),
        )

    @staticmethod
    def _ensure_exists(connection: sqlite3.Connection, consultation_id: str) -> None:
        row = connection.execute(
            "SELECT 1 FROM consultations WHERE id = ?", (consultation_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Consultation not found: {consultation_id}")
