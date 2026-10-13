import json

from travel_agent import cli
from travel_agent.repository import SQLiteRepository
from travel_agent.service import ConsultationService


def test_consultation_persists_explicit_customer_id(tmp_path):
    service = ConsultationService(SQLiteRepository(tmp_path / "travel.db"))

    consultation = service.create_consultation(
        "consultant-1",
        "张先生",
        "我想去杭州旅行",
        customer_id="customer-001",
    )

    restored = service.get_consultation(consultation.id)
    assert consultation.customer_id == "customer-001"
    assert restored.customer_id == "customer-001"


def test_legacy_consultation_gets_stable_compatibility_customer_id(tmp_path):
    service = ConsultationService(SQLiteRepository(tmp_path / "travel.db"))

    consultation = service.create_consultation("consultant-1", "李先生", "去杭州")

    assert consultation.customer_id == "legacy:李先生"
    assert service.get_consultation(consultation.id).customer_id == "legacy:李先生"


def test_cli_start_accepts_customer_id_and_prints_it(tmp_path, capsys):
    database = tmp_path / "travel.db"

    assert (
        cli.main(
            [
                "--database",
                str(database),
                "start",
                "--consultant-id",
                "consultant-1",
                "--customer-id",
                "customer-001",
                "--customer-name",
                "张先生",
                "--message",
                "我想去杭州旅行",
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["customer_id"] == "customer-001"
