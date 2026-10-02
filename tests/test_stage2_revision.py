from travel_agent.repository import SQLiteRepository
from travel_agent.service import ConsultationService


def complete_service(tmp_path):
    repository = SQLiteRepository(tmp_path / "travel_agent.db")
    service = ConsultationService(repository)
    consultation = service.create_consultation("consultant-1", "客户", "杭州旅行")
    service.update_requirements(
        consultation.id,
        {"origin":"北京", "destination":"杭州", "start_date":"2026-10-01",
         "end_date":"2026-10-04", "traveler_count":2, "budget":8000},
    )
    return repository, service, consultation


def test_revision_updates_constraints_and_creates_new_version(tmp_path):
    repository, service, consultation = complete_service(tmp_path)
    service.plan_consultation(consultation.id)

    result = service.revise_consultation(
        consultation.id, "预算调整为6000元，优先高铁，酒店换成评分更高的"
    )

    assert result["itinerary_version"] == 2
    restored = service.get_consultation(consultation.id)
    assert restored.trip_requirements["budget"] == 6000.0
    assert restored.planning_constraints == {
        "transport_mode": "train",
        "hotel_preference": "highest_rating",
    }
    assert result["itinerary"]["hotel_option_id"] == "hotel-hz-001"
    assert restored.messages[-1].content == "预算调整为6000元，优先高铁，酒店换成评分更高的"
    versions = repository.list_itinerary_versions(consultation.id)
    assert [version["version_number"] for version in versions] == [1, 2]


def test_revision_without_supported_constraint_still_records_message(tmp_path):
    repository, service, consultation = complete_service(tmp_path)

    result = service.revise_consultation(consultation.id, "请重新看看方案")

    assert result["status"] == "COMPLETED"
    assert service.get_consultation(consultation.id).messages[-1].content == "请重新看看方案"
