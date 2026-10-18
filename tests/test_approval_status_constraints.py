"""approval_requests / mock_bookings 的状态取值约束与旧库迁移。

故障背景：旧库 `approval_requests.status` 没有取值约束，一行 `status='1'`
被写坏后，读取时 `ApprovalStatus('1')` 抛 ValueError，整条审批链路卡死——
顾问既批不了也拒不了，只能手工改库。

SQLite 不支持 `ALTER TABLE ... ADD CONSTRAINT`，因此按官方建议重建表：
建新表（带 CHECK）→ 拷数据（非法值归一到安全默认）→ 换名。
"""

import sqlite3

import pytest

from travel_agent.domain import ApprovalDecision, ApprovalStatus, BookingStatus
from travel_agent.repository import SQLiteRepository


LEGACY_APPROVAL_SCHEMA = """
CREATE TABLE consultations (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    consultant_id TEXT NOT NULL,
    customer_name TEXT NOT NULL,
    status TEXT NOT NULL,
    trip_requirements TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    planning_constraints TEXT NOT NULL DEFAULT '{}',
    customer_id TEXT NOT NULL DEFAULT ''
);
CREATE TABLE approval_requests (
    id TEXT PRIMARY KEY,
    consultation_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    tool_arguments TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    status TEXT NOT NULL,
    requested_by TEXT NOT NULL,
    decision TEXT,
    decided_by TEXT,
    decision_reason TEXT,
    edited_arguments TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    expires_at TEXT,
    interrupt_id TEXT,
    action_index INTEGER
);
CREATE TABLE mock_bookings (
    id TEXT PRIMARY KEY,
    consultation_id TEXT NOT NULL,
    itinerary_version_id TEXT NOT NULL,
    total_amount REAL NOT NULL,
    status TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    cancellation_idempotency_key TEXT UNIQUE,
    cancellation_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    cancelled_at TEXT
);
"""


def legacy_database(path):
    connection = sqlite3.connect(path)
    connection.executescript(LEGACY_APPROVAL_SCHEMA)
    connection.execute(
        "INSERT INTO consultations (id,thread_id,consultant_id,customer_name,status,"
        "trip_requirements,created_at,updated_at) VALUES "
        "('c1','th1','consultant-1','客户','WAITING_FOR_APPROVAL','{}','2026-10-17','2026-10-17')"
    )
    connection.commit()
    connection.close()


def insert_approval(path, approval_id, status, decision=None, interrupt_id=None):
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO approval_requests (id,consultation_id,thread_id,tool_name,"
        "tool_arguments,risk_level,status,requested_by,decision,created_at,"
        "interrupt_id,action_index) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (approval_id, "c1", "th1", "create_mock_booking", '{"total_amount": 100}',
         "SENSITIVE", status, "agent", decision, "2026-10-17", interrupt_id, 0),
    )
    connection.commit()
    connection.close()


def schema_sql(path, table):
    connection = sqlite3.connect(path)
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    connection.close()
    return row[0] if row else ""


# --- 迁移 -------------------------------------------------------------------


def test_legacy_database_gets_check_constraints(tmp_path):
    path = tmp_path / "legacy.db"
    legacy_database(path)
    assert "CHECK" not in schema_sql(path, "approval_requests")

    SQLiteRepository(path)

    assert "CHECK" in schema_sql(path, "approval_requests")
    assert "CHECK" in schema_sql(path, "mock_bookings")


def test_migration_preserves_valid_rows(tmp_path):
    path = tmp_path / "legacy.db"
    legacy_database(path)
    insert_approval(path, "a-pending", "PENDING", interrupt_id="i1")
    insert_approval(path, "a-approved", "APPROVED", decision="APPROVE", interrupt_id="i2")
    insert_approval(path, "a-rejected", "REJECTED", decision="REJECT", interrupt_id="i3")

    repository = SQLiteRepository(path)

    statuses = {a.id: a.status for a in repository.list_approval_requests("c1")}
    assert statuses == {
        "a-pending": ApprovalStatus.PENDING,
        "a-approved": ApprovalStatus.APPROVED,
        "a-rejected": ApprovalStatus.REJECTED,
    }


def test_migration_repairs_invalid_status_instead_of_crashing(tmp_path):
    """旧库里 status='1' 的行会让读取抛枚举异常，迁移要把它救回来。"""
    path = tmp_path / "legacy.db"
    legacy_database(path)
    insert_approval(path, "a-broken", "1", interrupt_id="i1")

    repository = SQLiteRepository(path)

    approval = repository.get_approval_request("a-broken")
    assert approval.status is ApprovalStatus.PENDING


def test_migration_clears_invalid_decision(tmp_path):
    path = tmp_path / "legacy.db"
    legacy_database(path)
    insert_approval(path, "a-broken", "PENDING", decision="MAYBE", interrupt_id="i1")

    repository = SQLiteRepository(path)

    assert repository.get_approval_request("a-broken").decision is None


def test_migration_preserves_bookings(tmp_path):
    path = tmp_path / "legacy.db"
    legacy_database(path)
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO mock_bookings (id,consultation_id,itinerary_version_id,"
        "total_amount,status,idempotency_key,created_at,updated_at) "
        "VALUES ('b1','c1','v1',2790.0,'CREATED','k1','2026-10-17','2026-10-17')"
    )
    connection.commit()
    connection.close()

    repository = SQLiteRepository(path)

    bookings = repository.list_mock_bookings("c1")
    assert [b.id for b in bookings] == ["b1"]
    assert bookings[0].status is BookingStatus.CREATED


def test_migration_keeps_unique_interrupt_index(tmp_path):
    """重建表会丢掉旧索引，迁移必须把它补回来，否则幂等约束失效。"""
    path = tmp_path / "legacy.db"
    legacy_database(path)

    SQLiteRepository(path)

    connection = sqlite3.connect(path)
    indexes = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='approval_requests'"
        )
    }
    connection.close()
    assert "approval_requests_interrupt_action" in indexes


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "legacy.db"
    legacy_database(path)
    insert_approval(path, "a-pending", "PENDING", interrupt_id="i1")

    for _ in range(3):
        SQLiteRepository(path)

    repository = SQLiteRepository(path)
    assert len(repository.list_approval_requests("c1")) == 1


def test_fresh_database_is_created_with_constraints(tmp_path):
    path = tmp_path / "fresh.db"

    SQLiteRepository(path)

    assert "CHECK" in schema_sql(path, "approval_requests")
    assert "CHECK" in schema_sql(path, "mock_bookings")


# --- 约束生效 ---------------------------------------------------------------


def test_invalid_status_is_rejected_by_the_database(tmp_path):
    path = tmp_path / "t.db"
    SQLiteRepository(path)
    # 直接写库，绕过仓储层的枚举归一
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO consultations (id,thread_id,consultant_id,customer_name,status,"
        "trip_requirements,created_at,updated_at) VALUES "
        "('c1','th1','k','客','OPEN','{}','2026-10-17','2026-10-17')"
    )
    connection.commit()

    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        connection.execute(
            "INSERT INTO approval_requests (id,consultation_id,thread_id,tool_name,"
            "tool_arguments,risk_level,status,requested_by,created_at) "
            "VALUES ('a1','c1','th1','create_mock_booking','{}','SENSITIVE','1','agent','2026-10-17')"
        )
    connection.close()


def test_invalid_decision_is_rejected_by_the_database(tmp_path):
    path = tmp_path / "t.db"
    SQLiteRepository(path)
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO consultations (id,thread_id,consultant_id,customer_name,status,"
        "trip_requirements,created_at,updated_at) VALUES "
        "('c1','th1','k','客','OPEN','{}','2026-10-17','2026-10-17')"
    )
    connection.execute(
        "INSERT INTO approval_requests (id,consultation_id,thread_id,tool_name,"
        "tool_arguments,risk_level,status,requested_by,created_at) "
        "VALUES ('a1','c1','th1','create_mock_booking','{}','SENSITIVE','PENDING','agent','2026-10-17')"
    )
    connection.commit()

    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        connection.execute("UPDATE approval_requests SET decision='MAYBE' WHERE id='a1'")
    connection.close()


def test_repository_still_writes_valid_enum_values(tmp_path):
    path = tmp_path / "t.db"
    repository = SQLiteRepository(path)
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO consultations (id,thread_id,consultant_id,customer_name,status,"
        "trip_requirements,created_at,updated_at) VALUES "
        "('c1','th1','k','客','OPEN','{}','2026-10-17','2026-10-17')"
    )
    connection.commit()
    connection.close()

    approval = repository.create_approval_request(
        "c1", "th1", "create_mock_booking", {"total_amount": 100},
        requested_by="agent", interrupt_id="i1", action_index=0,
    )
    assert approval.status is ApprovalStatus.PENDING

    decided = repository.decide_approval_request(
        approval.id, ApprovalDecision.APPROVE, decided_by="consultant-1"
    )
    assert decided.status is ApprovalStatus.APPROVED
    assert decided.decision is ApprovalDecision.APPROVE
