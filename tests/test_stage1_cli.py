import json

from travel_agent.cli import main


def test_cli_start_and_status_show_persisted_consultation(tmp_path, capsys):
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

    assert main(common + ["status", created["consultation_id"]]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["consultation_id"] == created["consultation_id"]
    assert status["thread_id"] == created["thread_id"]
    assert status["status"] == "OPEN"
    assert status["messages"] == [{"role": "user", "content": "想去杭州"}]
    assert status["approval_requests"] == []


def test_cli_sensitive_tools_are_marked_for_human_approval(capsys):
    assert main(["sensitive-tools"]) == 0
    tools = json.loads(capsys.readouterr().out)

    assert [tool["name"] for tool in tools] == [
        "create_mock_booking",
        "cancel_mock_booking",
    ]
    assert all(tool["requires_human_approval"] is True for tool in tools)
    assert all(tool["permission"] == "SENSITIVE" for tool in tools)
