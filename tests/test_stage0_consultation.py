import sqlite3

from travel_agent.repository import SQLiteRepository
from travel_agent.service import ConsultationService


def test_create_consultation_persists_thread_and_audit_event(tmp_path):
    repository = SQLiteRepository(tmp_path / "travel_agent.db")
    service = ConsultationService(repository)

    consultation = service.create_consultation(
        consultant_id="consultant-1",
        customer_name="张三",
        initial_message="我想去杭州旅游",
    )

    restored = service.get_consultation(consultation.id)
    assert restored.id == consultation.id
    assert restored.customer_name == "张三"
    assert restored.thread_id
    assert restored.status == "OPEN"
    assert [message.as_dict() for message in restored.messages] == [
        {"role": "user", "content": "我想去杭州旅游"}
    ]

    events = repository.list_audit_events(consultation.id)
    assert [event.event_type for event in events] == ["CONSULTATION_CREATED"]


def test_append_message_updates_thread_and_creates_audit_event(tmp_path):
    repository = SQLiteRepository(tmp_path / "travel_agent.db")
    service = ConsultationService(repository)
    consultation = service.create_consultation("consultant-1", "李四", "想去成都")

    service.append_message(consultation.id, "assistant", "计划几天？")
    service.append_message(consultation.id, "user", "四天三晚")

    restored = service.get_consultation(consultation.id)
    assert [message.as_dict() for message in restored.messages] == [
        {"role": "user", "content": "想去成都"},
        {"role": "assistant", "content": "计划几天？"},
        {"role": "user", "content": "四天三晚"},
    ]
    events = repository.list_audit_events(consultation.id)
    assert [event.event_type for event in events] == [
        "CONSULTATION_CREATED",
        "MESSAGE_APPENDED",
        "MESSAGE_APPENDED",
    ]


def test_update_requirements_persists_structured_trip_requirements(tmp_path):
    repository = SQLiteRepository(tmp_path / "travel_agent.db")
    service = ConsultationService(repository)
    consultation = service.create_consultation("consultant-1", "王五", "想去三亚")

    service.update_requirements(
        consultation.id,
        {"origin": "北京", "destination": "三亚", "traveler_count": 2, "budget": 8000},
    )

    restored = service.get_consultation(consultation.id)
    assert restored.trip_requirements == {
        "origin": "北京",
        "destination": "三亚",
        "traveler_count": 2,
        "budget": 8000,
    }
    events = repository.list_audit_events(consultation.id)
    assert events[-1].event_type == "REQUIREMENTS_UPDATED"


def test_repository_uses_sqlite_tables(tmp_path):
    database_path = tmp_path / "travel_agent.db"
    SQLiteRepository(database_path)

    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert {"consultations", "messages", "audit_events"} <= tables
