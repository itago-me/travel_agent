from travel_agent.repository import SQLiteRepository
from travel_agent.service import ConsultationService


def complete_service(tmp_path):
    repository = SQLiteRepository(tmp_path / "travel_agent.db")
    service = ConsultationService(repository)
    consultation = service.create_consultation("consultant-1", "客户", "杭州旅行")
    service.update_requirements(
        consultation.id,
        {
            "origin": "北京", "destination": "杭州",
            "start_date": "2026-10-01", "end_date": "2026-10-04",
            "traveler_count": 2, "budget": 8000,
        },
    )
    return repository, service, consultation


def test_plan_persists_itinerary_draft_version_and_provider_requests(tmp_path):
    repository, service, consultation = complete_service(tmp_path)

    result = service.plan_consultation(consultation.id)
    versions = repository.list_itinerary_versions(consultation.id)
    requests = repository.list_provider_requests(consultation.id)

    assert result["itinerary_version"] == 1
    assert len(versions) == 1
    assert versions[0]["version_number"] == 1
    assert versions[0]["payload"]["budget"]["total"] == 3915.0
    assert len(requests) == 6
    assert requests[0]["tool_name"] == "search_transport_options"


def test_second_plan_creates_new_version_without_overwriting_first(tmp_path):
    repository, service, consultation = complete_service(tmp_path)

    service.plan_consultation(consultation.id)
    second = service.plan_consultation(consultation.id)
    versions = repository.list_itinerary_versions(consultation.id)

    assert second["itinerary_version"] == 2
    assert [version["version_number"] for version in versions] == [1, 2]
    assert versions[0]["id"] != versions[1]["id"]
