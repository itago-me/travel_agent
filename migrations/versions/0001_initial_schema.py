"""initial schema: 10 tables, 2 indexes

对应 ``repository._initialize()`` 中 SQLite 手写迁移的最终形态。

三处刻意的写法差异：

1. **去掉 ``IF NOT EXISTS``**：迁移只运行一次，带它会把「表已存在」这类
   真正的问题吞掉。Alembic 的 ``alembic_version`` 表负责判断是否已跑过。
2. **不搬 ``repository.py`` 里的 9 处 ``ALTER TABLE``**：那些是给老 SQLite
   库补列的历史补丁。这个迁移描述的是「全新库的最终形态」，直接建全量。
3. **CHECK 约束原样保留**：6 处都是标准 SQL，Postgres 与 SQLite 语义一致。

Revision ID: 0001
Revises:
Create Date: 2026-11-19
"""

from __future__ import annotations

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


# 建表顺序即外键依赖顺序：被引用的表必须先存在。
# consultation 是所有表的根；itinerary_drafts 必须先于 itinerary_versions。
_UPGRADE_STATEMENTS = [
    """CREATE TABLE consultations (
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
    )""",
    """CREATE TABLE messages (
        id TEXT PRIMARY KEY,
        consultation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (consultation_id) REFERENCES consultations(id)
    )""",
    """CREATE TABLE audit_events (
        id TEXT PRIMARY KEY,
        consultation_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        details TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (consultation_id) REFERENCES consultations(id)
    )""",
    """CREATE TABLE turn_spans (
        id TEXT PRIMARY KEY,
        consultation_id TEXT NOT NULL,
        turn_id TEXT NOT NULL,
        sequence INTEGER NOT NULL,
        kind TEXT NOT NULL,
        name TEXT NOT NULL,
        duration_ms INTEGER NOT NULL,
        details TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        FOREIGN KEY (consultation_id) REFERENCES consultations(id)
    )""",
    "CREATE INDEX idx_turn_spans_turn ON turn_spans(turn_id, sequence)",
    """CREATE TABLE itinerary_drafts (
        id TEXT PRIMARY KEY,
        consultation_id TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        selected_version_id TEXT,
        selected_at TEXT,
        FOREIGN KEY (consultation_id) REFERENCES consultations(id)
    )""",
    """CREATE TABLE itinerary_versions (
        id TEXT PRIMARY KEY,
        draft_id TEXT NOT NULL,
        version_number INTEGER NOT NULL,
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (draft_id, version_number),
        FOREIGN KEY (draft_id) REFERENCES itinerary_drafts(id)
    )""",
    """CREATE TABLE provider_requests (
        id TEXT PRIMARY KEY,
        consultation_id TEXT NOT NULL,
        version_id TEXT NOT NULL,
        tool_name TEXT NOT NULL,
        call_number INTEGER NOT NULL,
        response_payload TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (consultation_id) REFERENCES consultations(id),
        FOREIGN KEY (version_id) REFERENCES itinerary_versions(id)
    )""",
    """CREATE TABLE customer_preferences (
        customer_id TEXT NOT NULL,
        preference_key TEXT NOT NULL,
        preference_value TEXT NOT NULL,
        source TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (customer_id, preference_key)
    )""",
    """CREATE TABLE approval_requests (
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
    # 部分唯一索引：允许 interrupt_id 为 NULL 的多行，但同一 interrupt 的
    # 同一 action 只能有一条。SQLite 与 Postgres 都支持带 WHERE 的索引。
    """CREATE UNIQUE INDEX approval_requests_interrupt_action
           ON approval_requests(interrupt_id, action_index)
           WHERE interrupt_id IS NOT NULL""",
    """CREATE TABLE mock_bookings (
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
]

# 逆序删除：先删引用方，再删被引用方。索引随表一起消失。
_DOWNGRADE_TABLES = [
    "mock_bookings",
    "approval_requests",
    "customer_preferences",
    "provider_requests",
    "itinerary_versions",
    "itinerary_drafts",
    "turn_spans",
    "audit_events",
    "messages",
    "consultations",
]


def upgrade() -> None:
    for statement in _UPGRADE_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    for table in _DOWNGRADE_TABLES:
        op.execute(f"DROP TABLE {table}")
