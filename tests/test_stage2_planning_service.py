from travel_agent.repository import SQLiteRepository
from travel_agent.service import ConsultationService


def test_plan_consultation_runs_agent_tool_loop_and_audits_result(tmp_path):
    repository = SQLiteRepository(tmp_path / "travel_agent.db")
    service = ConsultationService(repository)
    consultation = service.create_consultation("consultant-1", "客户", "杭州旅行")
    service.update_requirements(
        consultation.id,
        {
            "origin": "北京",
            "destination": "杭州",
            "start_date": "2026-10-01",
            "end_date": "2026-10-04",
            "traveler_count": 2,
            "budget": 8000,
        },
    )

    result = service.plan_consultation(consultation.id)

    assert result["status"] == "COMPLETED"
    assert result["tool_call_count"] == 6
    assert result["itinerary"]["transport_option_id"] == "train-bj-hz-001"
    events = repository.list_audit_events(consultation.id)
    assert len([event for event in events if event.event_type == "TOOL_CALLED"]) == 6
    assert events[-1].event_type == "PLANNING_COMPLETED"
    assert events[-1].details["tool_call_count"] == 6
