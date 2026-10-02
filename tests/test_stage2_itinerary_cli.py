import json

from travel_agent.cli import main


def test_cli_itineraries_lists_saved_versions(tmp_path, capsys):
    database = tmp_path / "travel_agent.db"
    start_args = ["--database", str(database), "start", "--consultant-id", "c1", "--customer-name", "客户", "--message", "杭州"]
    main(start_args)
    consultation_id = json.loads(capsys.readouterr().out)["consultation_id"]
    from travel_agent.repository import SQLiteRepository
    from travel_agent.service import ConsultationService
    service = ConsultationService(SQLiteRepository(database))
    service.update_requirements(consultation_id, {"origin":"北京","destination":"杭州","start_date":"2026-10-01","end_date":"2026-10-04","traveler_count":2,"budget":8000})
    service.plan_consultation(consultation_id)

    assert main(["--database", str(database), "itineraries", consultation_id]) == 0
    versions = json.loads(capsys.readouterr().out)
    assert len(versions) == 1
    assert versions[0]["version_number"] == 1
