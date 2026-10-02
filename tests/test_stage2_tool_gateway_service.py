from travel_agent.repository import SQLiteRepository
from travel_agent.service import ConsultationService


def test_search_options_uses_gateway_and_persists_tool_audit_events(tmp_path):
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

    result = service.search_options(consultation.id)

    assert len(result["transport"]["options"]) == 2
    assert len(result["hotels"]["options"]) == 2
    assert len(result["attractions"]["options"]) == 2
    tool_events = [
        event
        for event in repository.list_audit_events(consultation.id)
        if event.event_type == "TOOL_CALLED"
    ]
    assert [event.details["tool_name"] for event in tool_events] == [
        "search_transport_options",
        "search_hotel_options",
        "search_attractions",
    ]
    assert all(event.details["outcome"] == "SUCCEEDED" for event in tool_events)


def test_search_options_rejects_incomplete_requirements(tmp_path):
    service = ConsultationService(SQLiteRepository(tmp_path / "travel_agent.db"))
    consultation = service.create_consultation("consultant-1", "客户", "想去杭州")

    try:
        service.search_options(consultation.id)
    except ValueError as error:
        assert "旅行需求不完整" in str(error)
    else:
        raise AssertionError("incomplete requirements were accepted")
