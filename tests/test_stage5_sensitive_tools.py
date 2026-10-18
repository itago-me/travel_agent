import asyncio
import json

import pytest
from pydantic import ValidationError

from travel_agent.booking import (
    CancelMockBookingRequest,
    CreateMockBookingRequest,
)
from travel_agent.cli import main
from travel_agent.langchain_tools import create_sensitive_booking_tools
from travel_agent.repository import SQLiteRepository
from travel_agent.service import ConsultationService


def prepared(tmp_path):
    database = tmp_path / "travel.db"
    repository = SQLiteRepository(database)
    service = ConsultationService(repository)
    consultation = service.create_consultation("c1", "客户", "杭州")
    version_id, _ = repository.save_itinerary_version(
        consultation.id, {"budget": {"total": 7800}}, [], {}
    )
    repository.select_itinerary_version(consultation.id, version_id)
    return database, repository, consultation, version_id


def test_sensitive_schemas_reject_invalid_money_and_empty_ids():
    with pytest.raises(ValidationError):
        CreateMockBookingRequest(
            consultation_id="",
            itinerary_version_id="version-1",
            total_amount=-1,
            idempotency_key="key-1",
        )
    with pytest.raises(ValidationError):
        CancelMockBookingRequest(
            consultation_id="consultation-1",
            booking_id="",
            reason="取消",
            idempotency_key="key-2",
        )


def test_sensitive_tools_have_schema_permission_and_structured_results(tmp_path):
    database, repository, consultation, version_id = prepared(tmp_path)
    tools = {tool.name: tool for tool in create_sensitive_booking_tools(database)}
    assert set(tools) == {"create_mock_booking", "cancel_mock_booking"}
    assert all(tool.metadata["permission"] == "SENSITIVE" for tool in tools.values())

    created = asyncio.run(tools["create_mock_booking"].ainvoke({
        "consultation_id": consultation.id,
        "itinerary_version_id": version_id,
        "total_amount": 7800,
        "idempotency_key": "booking-tool-1",
    }))
    assert created["status"] == "CREATED"
    assert len(repository.list_mock_bookings(consultation.id)) == 1

    cancelled = asyncio.run(tools["cancel_mock_booking"].ainvoke({
        "consultation_id": consultation.id,
        "booking_id": created["booking_id"],
        "reason": "客户取消",
        "idempotency_key": "cancel-tool-1",
    }))
    assert cancelled["status"] == "CANCELLED"


def test_default_agent_tools_do_not_expose_sensitive_tools_before_hitl(tmp_path):
    from travel_agent.langchain_tools import create_travel_tools

    names = {tool.name for tool in create_travel_tools(tmp_path / "travel.db")}
    assert "create_mock_booking" not in names
    assert "cancel_mock_booking" not in names


def test_create_booking_rejects_an_unselected_itinerary(tmp_path):
    database = tmp_path / "travel.db"
    repository = SQLiteRepository(database)
    service = ConsultationService(repository)
    consultation = service.create_consultation("c1", "客户", "杭州")
    version_id, _ = repository.save_itinerary_version(
        consultation.id, {"budget": {"total": 7800}}, [], {}
    )
    tool = {tool.name: tool for tool in create_sensitive_booking_tools(database)}[
        "create_mock_booking"
    ]
    with pytest.raises(ValueError, match="selected"):
        asyncio.run(tool.ainvoke({
            "consultation_id": consultation.id,
            "itinerary_version_id": version_id,
            "total_amount": 7800,
            "idempotency_key": "unselected-1",
        }))


def test_cli_lists_sensitive_tools_without_executing_them(tmp_path, capsys):
    assert main([
        "--database", str(tmp_path / "travel.db"), "sensitive-tools"
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {item["name"] for item in payload} == {
        "create_mock_booking", "cancel_mock_booking"
    }
    assert {item["permission"] for item in payload} == {"SENSITIVE"}
