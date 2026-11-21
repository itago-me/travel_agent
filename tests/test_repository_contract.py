"""阶段 9 B3：仓储契约测试——同一套断言跑两种后端。

这是「两个实现行为一致」的唯一证明。每个用例都只通过 ``RepositoryProtocol``
的方法交互，不碰实现细节；``repository`` fixture 决定跑哪个后端。

没有 Postgres 时，``postgres`` 参数会 skip，sqlite 参数照常跑——默认
`pytest` 不依赖数据库服务这条基线不变。

几个刻意覆盖的点：

- **``ON CONFLICT`` 的 upsert 语义**：SQLite 与 Postgres 语法相同，但
  ``excluded`` 的取值行为要实测确认。
- **``rowcount``**：删除/更新的影响行数在两种驱动下的语义一致。
- **``REAL`` 与 ``float`` 的往返**：SQLite 的 REAL 和 Postgres 的 float4
  在精度上可能有差异，金额比较用近似。
- **异常类型**：未知 id 必须抛 ``KeyError``（不是驱动的 IntegrityError），
  否则 API 层翻译 404 的逻辑会失效。
"""

from __future__ import annotations

import pytest

from travel_agent.domain import (
    ApprovalDecision,
    ApprovalStatus,
    AuditEvent,
    BookingStatus,
    Consultation,
    Message,
    utc_now,
)
from travel_agent.postgres_repository import PostgresRepository
from travel_agent.repository import SQLiteRepository
from conftest import postgres_url, plain_url


def _consultation(**overrides) -> Consultation:
    from uuid import uuid4

    fields = {
        "id": str(uuid4()),
        "thread_id": str(uuid4()),
        "consultant_id": "consultant-1",
        "customer_id": "customer-1",
        "customer_name": "张三",
        "status": "OPEN",
        "trip_requirements": {"destination": "杭州", "traveler_count": 2},
    }
    fields.update(overrides)
    return Consultation(**fields)


@pytest.fixture(params=["sqlite", "postgres"])
def repository(request, tmp_path):
    """同一个 fixture 产出两种实现；Postgres 不可用时该参数 skip。"""
    if request.param == "sqlite":
        yield SQLiteRepository(tmp_path / "contract.db")
        return

    url = postgres_url()
    if url is None:
        pytest.skip("未设置 TRAVEL_AGENT_TEST_DATABASE_URL")
    try:
        import psycopg
        from psycopg.rows import dict_row

        from alembic import command

        from conftest import _alembic_config, drop_schema

        admin = psycopg.connect(plain_url(url), autocommit=True, row_factory=dict_row)
    except ImportError:
        pytest.skip("未安装 psycopg / alembic")

    from uuid import uuid4

    schema = f"test_{uuid4().hex[:12]}"
    admin.execute(f"CREATE SCHEMA {schema}")
    try:
        command.upgrade(_alembic_config(url, schema), "head")
        dsn = plain_url(url)
        repo = PostgresRepository(dsn)
        # 让这个仓储的所有连接都落到测试 schema
        repo.close()
        repo = PostgresRepository(dsn, schema=schema)
        try:
            yield repo
        finally:
            repo.close()
    finally:
        drop_schema(admin, schema)
        admin.close()


def test_consultation_roundtrip(repository):
    entity = _consultation()
    repository.create_consultation(entity)

    loaded = repository.get_consultation(entity.id)

    assert loaded.id == entity.id
    assert loaded.thread_id == entity.thread_id
    assert loaded.customer_name == "张三"
    assert loaded.trip_requirements == {"destination": "杭州", "traveler_count": 2}
    assert loaded.messages == []


def test_unknown_consultation_raises_key_error(repository):
    """必须是 KeyError——API 层靠它翻译成 404。"""
    with pytest.raises(KeyError):
        repository.get_consultation("no-such-id")


def test_messages_are_appended_in_order(repository):
    entity = _consultation()
    repository.create_consultation(entity)
    repository.append_message(entity.id, Message("user", "想去杭州"))
    repository.append_message(entity.id, Message("assistant", "好的"))

    loaded = repository.get_consultation(entity.id)

    assert [m.role for m in loaded.messages] == ["user", "assistant"]
    assert [m.content for m in loaded.messages] == ["想去杭州", "好的"]


def test_requirements_update_is_persisted(repository):
    entity = _consultation()
    repository.create_consultation(entity)

    repository.update_requirements(entity.id, {"budget": 8000.0})

    assert repository.get_consultation(entity.id).trip_requirements == {"budget": 8000.0}


def test_status_update_is_persisted(repository):
    entity = _consultation()
    repository.create_consultation(entity)

    repository.update_consultation_status(entity.id, "WAITING_FOR_APPROVAL")

    assert repository.get_consultation(entity.id).status == "WAITING_FOR_APPROVAL"


def test_audit_events_roundtrip(repository):
    entity = _consultation()
    repository.create_consultation(entity)
    repository.add_audit_event(
        AuditEvent(entity.id, "TOOL_EXECUTED", {"tool_name": "search_hotel"})
    )

    events = repository.list_audit_events(entity.id)

    assert len(events) == 1
    assert events[0].event_type == "TOOL_EXECUTED"
    assert events[0].details["tool_name"] == "search_hotel"
    assert repository.count_tool_executions(entity.id) == 1


def test_turn_spans_keep_sequence_order(repository):
    """并行工具的插入时刻交错，排序只能靠 sequence。"""
    entity = _consultation()
    repository.create_consultation(entity)
    for sequence in (1, 3, 2):
        repository.add_turn_span(
            consultation_id=entity.id,
            turn_id="turn-1",
            kind="tool",
            name=f"tool-{sequence}",
            duration_ms=sequence,
            sequence=sequence,
        )

    spans = repository.list_turn_spans(entity.id)

    assert [s["sequence"] for s in spans] == [1, 2, 3]
    assert [s["name"] for s in spans] == ["tool-1", "tool-2", "tool-3"]


def test_customer_preference_upsert(repository):
    """``ON CONFLICT DO UPDATE`` 在两种后端下都要覆盖旧值而不是插两行。"""
    repository.set_customer_preference("cust-1", "seat", "靠窗")
    repository.set_customer_preference("cust-1", "seat", "靠过道", source="explicit")

    preferences = repository.list_customer_preferences("cust-1")

    assert len(preferences) == 1
    assert preferences[0].value == "靠过道"


def test_delete_preference_reports_rowcount(repository):
    repository.set_customer_preference("cust-1", "seat", "靠窗")

    assert repository.delete_customer_preference("cust-1", "seat") is True
    assert repository.delete_customer_preference("cust-1", "seat") is False


def test_itinerary_version_roundtrip(repository):
    entity = _consultation()
    repository.create_consultation(entity)

    version_id, version_number = repository.save_itinerary_version(
        entity.id, {"budget": {"total": 2790.0}}, ["search_hotel"], {}
    )

    versions = repository.list_itinerary_versions(entity.id)
    assert version_number == 1
    assert len(versions) == 1
    assert versions[0]["payload"]["budget"]["total"] == 2790.0

    loaded = repository.get_itinerary_version(entity.id, version_id)
    assert loaded["version_number"] == 1


def test_unknown_version_raises_key_error(repository):
    entity = _consultation()
    repository.create_consultation(entity)

    with pytest.raises(KeyError):
        repository.get_itinerary_version(entity.id, "no-such-version")


def test_select_itinerary_version(repository):
    entity = _consultation()
    repository.create_consultation(entity)
    version_id, _ = repository.save_itinerary_version(entity.id, {}, [], {})

    repository.select_itinerary_version(entity.id, version_id)

    selected = repository.get_selected_itinerary(entity.id)
    assert selected is not None
    assert selected["version_id"] == version_id


def test_booking_requires_selected_version(repository):
    entity = _consultation()
    repository.create_consultation(entity)
    version_id, _ = repository.save_itinerary_version(
        entity.id, {"budget": {"total": 1000.0}}, [], {}
    )

    with pytest.raises(ValueError, match="must be selected"):
        repository.create_mock_booking(
            entity.id, version_id, total_amount=1000.0, idempotency_key="k1"
        )


def test_booking_roundtrip_and_idempotency(repository):
    entity = _consultation()
    repository.create_consultation(entity)
    version_id, _ = repository.save_itinerary_version(
        entity.id, {"budget": {"total": 1000.0}}, [], {}
    )
    repository.select_itinerary_version(entity.id, version_id)

    booking = repository.create_mock_booking(
        entity.id, version_id, total_amount=1000.0, idempotency_key="k1"
    )
    replay = repository.create_mock_booking(
        entity.id, version_id, total_amount=1000.0, idempotency_key="k1"
    )

    assert booking.id == replay.id
    assert booking.status == BookingStatus.CREATED
    assert float(booking.total_amount) == pytest.approx(1000.0)
    assert len(repository.list_mock_bookings(entity.id)) == 1


def test_booking_cancellation(repository):
    entity = _consultation()
    repository.create_consultation(entity)
    version_id, _ = repository.save_itinerary_version(
        entity.id, {"budget": {"total": 1000.0}}, [], {}
    )
    repository.select_itinerary_version(entity.id, version_id)
    booking = repository.create_mock_booking(
        entity.id, version_id, total_amount=1000.0, idempotency_key="k1"
    )

    cancelled = repository.cancel_mock_booking(
        entity.id, booking.id, reason="客户改期", idempotency_key="cancel-1"
    )

    assert cancelled.status == BookingStatus.CANCELLED
    assert cancelled.cancellation_reason == "客户改期"


def test_approval_decision_roundtrip(repository):
    entity = _consultation()
    repository.create_consultation(entity)
    approval = repository.create_approval_request(
        entity.id,
        entity.thread_id,
        "create_mock_booking",
        {"total_amount": 100},
        requested_by="agent",
    )

    assert approval.status == ApprovalStatus.PENDING
    assert len(repository.list_pending_approval_requests(entity.id)) == 1

    decided = repository.decide_approval_request(
        approval.id, ApprovalDecision.APPROVE, decided_by="顾问李四"
    )

    assert decided.status == ApprovalStatus.APPROVED
    assert decided.decided_by == "顾问李四"
    assert repository.list_pending_approval_requests(entity.id) == []


def test_deciding_twice_is_rejected(repository):
    entity = _consultation()
    repository.create_consultation(entity)
    approval = repository.create_approval_request(
        entity.id, entity.thread_id, "create_mock_booking", {}, requested_by="agent"
    )
    repository.decide_approval_request(
        approval.id, ApprovalDecision.APPROVE, decided_by="顾问"
    )

    with pytest.raises(ValueError, match="already decided"):
        repository.decide_approval_request(
            approval.id, ApprovalDecision.REJECT, decided_by="顾问", reason="反悔"
        )


def test_usage_and_metrics_aggregate_the_audit_trail(repository):
    entity = _consultation()
    repository.create_consultation(entity)
    repository.add_audit_event(
        AuditEvent(
            entity.id,
            "AGENT_CHAT_COMPLETED",
            {
                "tool_calls": ["a", "b"],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "reported": True,
                },
                "duration_ms": 1200,
            },
        )
    )

    usage = repository.consult_usage_totals(entity.id)
    assert usage["turns"] == 1
    assert usage["total_tokens"] == 120

    metrics = repository.metrics_snapshot()
    assert metrics["turns"]["completed"] == 1
    assert metrics["latency_ms"]["max"] == 1200
    assert metrics["tools"]["per_turn"] == 2.0
