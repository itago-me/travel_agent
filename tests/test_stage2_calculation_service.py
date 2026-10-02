from travel_agent.repository import SQLiteRepository
from travel_agent.service import ConsultationService


def test_calculate_options_returns_budget_conflicts_and_score(tmp_path):
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

    result = service.calculate_options(consultation.id)

    assert result["budget"]["total"] == 3915.0
    assert result["budget"]["within_budget"] is True
    assert result["conflicts"]["has_conflicts"] is True
    assert len(result["conflicts"]["conflicts"]) == 1
    assert 0 <= result["score"]["score"] <= 100
    compute_events = [
        event for event in repository.list_audit_events(consultation.id)
        if event.event_type == "TOOL_CALLED"
        and event.details["permission"] == "COMPUTE"
    ]
    assert [event.details["tool_name"] for event in compute_events] == [
        "calculate_trip_budget",
        "detect_itinerary_conflicts",
        "score_trip_options",
    ]
