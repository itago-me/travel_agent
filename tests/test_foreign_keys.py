"""阶段 9 A1：外键行为基线。

换库前必须回答一个问题：**哪些代码依赖「外键不生效」？**

实测结论：**零处**。所有公开写入 API 都在事务开头调用 ``_ensure_exists``
（或等价的归属校验），孤儿行在到达数据库之前就被 ``KeyError`` 拦住。

但这层保护**完全来自应用代码**——SQLite 本身不强制外键（``PRAGMA
foreign_keys`` 是每连接设置且默认关闭，``_connect()`` 没有启用它）。
这个测试把两件事固定下来：

1. **基线事实**：数据库层确实不拦。绕过 repository 直接写 SQL，孤儿行能进去。
   这是换到 Postgres 后会改变的行为，先记录在案。
2. **回归保护**：应用层每个写入路径都必须拦住未知的 consultation。
   将来新增写入方法时忘了加 guard，这里的参数化测试会失败。

Postgres 切换后，第 1 组断言需要翻转（它开始强制外键），第 2 组不变。
"""

from uuid import uuid4

import pytest

from travel_agent.domain import AuditEvent, Consultation, Message
from travel_agent.repository import SQLiteRepository


def _consultation() -> Consultation:
    return Consultation(
        id=str(uuid4()),
        thread_id=str(uuid4()),
        consultant_id="consultant-1",
        customer_name="张三",
        status="OPEN",
        trip_requirements={},
    )


@pytest.fixture
def repository(tmp_path):
    return SQLiteRepository(tmp_path / "fk.db")


@pytest.fixture
def consultation(repository):
    entity = _consultation()
    repository.create_consultation(entity)
    return entity


# ---------------------------------------------------------------------------
# 组 1：数据库层不强制外键（基线事实）
# ---------------------------------------------------------------------------
#
# 这组断言描述的是 SQLite 的当前行为，不是我们希望的行为。
# 换到 Postgres 后它们会变成 IntegrityError —— 那时这组测试应当翻转成
# 「数据库层也会拦住」，作为迁移完成的证据。


def test_sqlite_connection_does_not_enable_foreign_keys(repository):
    """``_connect()`` 不启用 PRAGMA foreign_keys，每连接都是关闭状态。"""
    with repository._connect() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 0


def test_database_layer_accepts_orphan_rows(repository, consultation):
    """绕过应用层直接写 SQL，引用不存在父行的子行可以插入。

    这是迁移前必须知道的事实：数据库层没有兜底，唯一的防线是应用代码。
    """
    with repository._connect() as connection:
        connection.execute(
            """INSERT INTO messages (id, consultation_id, role, content, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            ("orphan-1", "no-such-consultation", "user", "hi", "2026-01-01T00:00:00"),
        )

    with repository._connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM messages WHERE consultation_id = 'no-such-consultation'"
        ).fetchone()[0]
    assert count == 1


# ---------------------------------------------------------------------------
# 组 2：应用层拦住每一个写入路径（回归保护）
# ---------------------------------------------------------------------------

GHOST = "this-consultation-does-not-exist"


def _write_paths():
    """每个公开写入 API，用一个不存在的 consultation_id 调用一次。

    新增写入方法时应当加进这个列表——否则「忘了加 guard」不会被发现。
    """
    return [
        ("append_message", lambda r: r.append_message(GHOST, Message("user", "hi"))),
        ("add_audit_event", lambda r: r.add_audit_event(AuditEvent(GHOST, "X", {}))),
        (
            "add_turn_span",
            lambda r: r.add_turn_span(
                consultation_id=GHOST,
                turn_id="t1",
                kind="model",
                name="模型调用 #1",
                duration_ms=1,
            ),
        ),
        ("save_itinerary_version", lambda r: r.save_itinerary_version(GHOST, {}, [], {})),
        (
            "create_approval_request",
            lambda r: r.create_approval_request(
                GHOST, "any-thread", "create_mock_booking", {}, requested_by="agent"
            ),
        ),
        (
            "create_mock_booking",
            lambda r: r.create_mock_booking(
                GHOST, "any-version", total_amount=100, idempotency_key="k1"
            ),
        ),
        ("update_requirements", lambda r: r.update_requirements(GHOST, {})),
        (
            "update_consultation_status",
            lambda r: r.update_consultation_status(GHOST, "OPEN"),
        ),
    ]


@pytest.mark.parametrize(
    "name,write",
    _write_paths(),
    ids=[name for name, _ in _write_paths()],
)
def test_write_path_rejects_unknown_consultation(repository, name, write):
    """未知 consultation 必须先抛 KeyError，绝不允许写进孤儿行。"""
    with pytest.raises(KeyError):
        write(repository)


def test_no_orphan_rows_after_all_write_paths(repository):
    """跑完所有被拒绝的写入路径，子表应当仍是空的。"""
    for _name, write in _write_paths():
        with pytest.raises(KeyError):
            write(repository)

    child_tables = (
        "messages",
        "audit_events",
        "turn_spans",
        "itinerary_drafts",
        "itinerary_versions",
        "provider_requests",
        "approval_requests",
        "mock_bookings",
    )
    with repository._connect() as connection:
        for table in child_tables:
            total = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert total == 0, f"{table} 出现了 {total} 条孤儿行"


# ---------------------------------------------------------------------------
# 组 3：二级外键（version_id）也做了归属校验
# ---------------------------------------------------------------------------


def test_mock_booking_rejects_version_from_nowhere(repository, consultation):
    """父行存在、但版本号不存在——必须被拦住，而不是写入孤儿 booking。"""
    with pytest.raises(KeyError):
        repository.create_mock_booking(
            consultation.id,
            "no-such-version",
            total_amount=100,
            idempotency_key="k1",
        )
