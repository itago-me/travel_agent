from travel_agent.repository import SQLiteRepository
from travel_agent.service import ConsultationService


def prepared(tmp_path):
    repository = SQLiteRepository(tmp_path / "travel_agent.db")
    service = ConsultationService(repository)
    consultation = service.create_consultation("c1", "客户", "杭州")
    service.update_requirements(consultation.id, {
        "origin":"北京", "destination":"杭州", "start_date":"2026-10-01",
        "end_date":"2026-10-04", "traveler_count":2, "budget":8000,
    })
    service.plan_consultation(consultation.id)
    service.revise_consultation(consultation.id, "预算调整为6000元")
    return repository, service, consultation


def test_compare_itineraries_returns_version_comparison(tmp_path):
    repository, service, consultation = prepared(tmp_path)

    comparison = service.compare_itineraries(consultation.id)

    assert [item["version_number"] for item in comparison] == [1, 2]
    assert comparison[0]["budget_limit"] == 8000.0
    assert comparison[1]["budget_limit"] == 6000.0
    assert comparison[0]["version_id"] != comparison[1]["version_id"]


def test_select_itinerary_persists_selection_and_audits_it(tmp_path):
    repository, service, consultation = prepared(tmp_path)
    versions = repository.list_itinerary_versions(consultation.id)
    selected_id = versions[1]["id"]

    selected = service.select_itinerary(consultation.id, selected_id)

    assert selected["version_id"] == selected_id
    assert selected["version_number"] == 2
    assert repository.get_selected_itinerary(consultation.id)["version_id"] == selected_id
    events = repository.list_audit_events(consultation.id)
    assert events[-1].event_type == "ITINERARY_SELECTED"
    assert events[-1].details["version_id"] == selected_id


def test_select_itinerary_rejects_unknown_version(tmp_path):
    repository, service, consultation = prepared(tmp_path)

    try:
        service.select_itinerary(consultation.id, "missing-version")
    except KeyError as error:
        assert "Itinerary version not found" in str(error)
    else:
        raise AssertionError("unknown itinerary version was accepted")
