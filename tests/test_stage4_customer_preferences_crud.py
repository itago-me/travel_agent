import json

import pytest

from travel_agent import cli
from travel_agent.repository import SQLiteRepository


def test_repository_creates_lists_and_gets_customer_preferences(tmp_path):
    repository = SQLiteRepository(tmp_path / "travel.db")

    hotel = repository.set_customer_preference(
        "customer-001", "hotel.star_level", "舒适型", source="explicit"
    )
    budget = repository.set_customer_preference(
        "customer-001", "budget.daily_limit", 800, source="explicit"
    )
    repository.set_customer_preference(
        "customer-002", "hotel.star_level", "豪华型", source="explicit"
    )

    assert hotel.customer_id == "customer-001"
    assert hotel.value == "舒适型"
    assert hotel.created_at == hotel.updated_at
    assert repository.get_customer_preference(
        "customer-001", "hotel.star_level"
    ) == hotel
    assert repository.list_customer_preferences("customer-001") == [budget, hotel]


def test_repository_updates_value_without_replacing_creation_time(tmp_path, monkeypatch):
    repository = SQLiteRepository(tmp_path / "travel.db")
    timestamps = iter(
        ["2026-10-10T01:00:00+00:00", "2026-10-10T02:00:00+00:00"]
    )
    monkeypatch.setattr("travel_agent.repository.utc_now", lambda: next(timestamps))

    original = repository.set_customer_preference(
        "customer-001", "transport.seat_class", "二等座", source="explicit"
    )
    updated = repository.set_customer_preference(
        "customer-001", "transport.seat_class", "一等座", source="confirmed"
    )

    assert updated.value == "一等座"
    assert updated.source == "confirmed"
    assert updated.created_at == original.created_at
    assert updated.updated_at == "2026-10-10T02:00:00+00:00"


def test_repository_deletes_only_requested_customer_preference(tmp_path):
    repository = SQLiteRepository(tmp_path / "travel.db")
    repository.set_customer_preference(
        "customer-001", "hotel.star_level", "舒适型", source="explicit"
    )
    repository.set_customer_preference(
        "customer-002", "hotel.star_level", "豪华型", source="explicit"
    )

    assert repository.delete_customer_preference(
        "customer-001", "hotel.star_level"
    ) is True
    assert repository.delete_customer_preference(
        "customer-001", "hotel.star_level"
    ) is False
    assert repository.get_customer_preference(
        "customer-001", "hotel.star_level"
    ) is None
    assert repository.get_customer_preference(
        "customer-002", "hotel.star_level"
    ).value == "豪华型"


def test_repository_rejects_empty_preference_identity(tmp_path):
    repository = SQLiteRepository(tmp_path / "travel.db")

    with pytest.raises(ValueError, match="customer_id"):
        repository.set_customer_preference("", "hotel.star_level", "舒适型")
    with pytest.raises(ValueError, match="preference_key"):
        repository.set_customer_preference("customer-001", "", "舒适型")


def test_cli_can_set_list_update_and_delete_customer_preference(tmp_path, capsys):
    database = tmp_path / "travel.db"
    common = ["--database", str(database)]

    assert cli.main(
        common
        + [
            "set-preference",
            "customer-001",
            "hotel.star_level",
            "--value-json",
            '"舒适型"',
        ]
    ) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["preference_key"] == "hotel.star_level"
    assert created["value"] == "舒适型"
    assert created["source"] == "explicit"

    assert cli.main(
        common
        + [
            "set-preference",
            "customer-001",
            "hotel.star_level",
            "--value-json",
            '"豪华型"',
            "--source",
            "confirmed",
        ]
    ) == 0
    updated = json.loads(capsys.readouterr().out)
    assert updated["value"] == "豪华型"
    assert updated["source"] == "confirmed"

    assert cli.main(common + ["preferences", "customer-001"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed == [updated]

    assert cli.main(
        common + ["delete-preference", "customer-001", "hotel.star_level"]
    ) == 0
    deleted = json.loads(capsys.readouterr().out)
    assert deleted == {
        "customer_id": "customer-001",
        "preference_key": "hotel.star_level",
        "deleted": True,
    }

    assert cli.main(common + ["preferences", "customer-001"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_cli_rejects_invalid_preference_json(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "--database",
                str(tmp_path / "travel.db"),
                "set-preference",
                "customer-001",
                "hotel.star_level",
                "--value-json",
                "not-json",
            ]
        )

    assert error.value.code == 2
    assert "必须是有效 JSON" in capsys.readouterr().err
