from travel_agent.repository import SQLiteRepository
from travel_agent.service import ConsultationService


def test_resume_persists_user_message_requirements_and_agent_reply(tmp_path):
    service = ConsultationService(SQLiteRepository(tmp_path / "travel_agent.db"))
    consultation = service.create_consultation("consultant-1", "客户", "想去杭州")

    result = service.resume_consultation(
        consultation.id,
        "从北京出发，去杭州，2026-10-01到2026-10-04，2人，预算8000元",
    )

    restored = service.get_consultation(consultation.id)
    assert result["decision"]["action"] == "READY"
    assert restored.trip_requirements == {
        "origin": "北京",
        "destination": "杭州",
        "start_date": "2026-10-01",
        "end_date": "2026-10-04",
        "traveler_count": 2,
        "budget": 8000.0,
    }
    assert restored.messages[-1].role == "assistant"
    assert restored.messages[-1].content == "旅行需求已收集完整，可以开始生成方案。"
    decision_events = [
        event for event in service.repository.list_audit_events(consultation.id)
        if event.event_type == "AGENT_DECISION"
    ]
    assert decision_events[-1].details["model_call_count"] == 1


def test_resume_keeps_checkpoint_call_count_for_the_same_thread(tmp_path):
    service = ConsultationService(SQLiteRepository(tmp_path / "travel_agent.db"))
    consultation = service.create_consultation("consultant-1", "客户", "想去杭州")

    first = service.resume_consultation(consultation.id, "从北京出发，去杭州")
    second = service.resume_consultation(consultation.id, "预算调整为8000元")

    assert first["model_call_count"] == 1
    assert second["model_call_count"] == 2
    assert second["requirements"]["origin"] == "北京"
    assert second["requirements"]["budget"] == 8000.0
