import json

from travel_agent.cli import main


def test_cli_start_resume_and_status_show_persisted_agent_workflow(
    tmp_path, monkeypatch, capsys
):
    database = tmp_path / "travel_agent.sqlite"
    checkpoint_database = tmp_path / "travel_agent_checkpoints.sqlite"
    common = ["--database", str(database), "--checkpoint-database", str(checkpoint_database)]

    assert main(
        common
        + [
            "start",
            "--consultant-id",
            "consultant-1",
            "--customer-name",
            "测试客户",
            "--message",
            "想去杭州",
        ]
    ) == 0
    created = json.loads(capsys.readouterr().out)

    assert main(
        common
        + [
            "resume",
            created["consultation_id"],
            "--message",
            "从北京出发，去杭州，2026-10-01到2026-10-04，2人，预算8000元",
        ]
    ) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["thread_id"] == created["thread_id"]
    assert resumed["decision"]["action"] == "READY"

    assert main(common + ["status", created["consultation_id"]]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["trip_requirements"]["origin"] == "北京"
    assert status["messages"][-1]["role"] == "assistant"


def test_cli_lists_tools(capsys):
    assert main(["tools"]) == 0
    tools = json.loads(capsys.readouterr().out)

    assert [tool["name"] for tool in tools] == [
        "calculate_trip_budget",
        "detect_itinerary_conflicts",
        "score_trip_options",
        "search_attractions",
        "search_hotel_options",
        "search_transport_options",
    ]
