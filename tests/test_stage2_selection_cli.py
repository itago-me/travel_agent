import json

from travel_agent.cli import main
from travel_agent.repository import SQLiteRepository
from travel_agent.service import ConsultationService


def test_cli_compare_and_select_itinerary(tmp_path, capsys):
    database = tmp_path / "travel_agent.db"
    main(["--database", str(database), "start", "--consultant-id", "c1", "--customer-name", "客户", "--message", "杭州"])
    consultation_id = json.loads(capsys.readouterr().out)["consultation_id"]
    service = ConsultationService(SQLiteRepository(database))
    service.update_requirements(consultation_id, {"origin":"北京","destination":"杭州","start_date":"2026-10-01","end_date":"2026-10-04","traveler_count":2,"budget":8000})
    service.plan_consultation(consultation_id)
    service.revise_consultation(consultation_id, "预算调整为6000元")
    versions = service.repository.list_itinerary_versions(consultation_id)

    assert main(["--database", str(database), "compare-itineraries", consultation_id]) == 0
    comparison = json.loads(capsys.readouterr().out)
    assert [item["version_number"] for item in comparison] == [1, 2]

    assert main(["--database", str(database), "select-itinerary", consultation_id, versions[1]["id"]]) == 0
    selected = json.loads(capsys.readouterr().out)
    assert selected["version_number"] == 2
