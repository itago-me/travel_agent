import asyncio
import json
from typing import Any

import pytest
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from travel_agent.agent import AgentSettings, create_agent_app
from travel_agent.application import create_consultation_service
from travel_agent.cli import main
from travel_agent.domain import ApprovalStatus, BookingStatus
from travel_agent.sqlite_providers import initialize_travel_data


class BookingToolCallingModel(BaseChatModel):
    consultation_id: str
    itinerary_version_id: str
    invocations: int = 0
    requested_actions: list[dict[str, Any]] | None = None

    @property
    def _llm_type(self):
        return "offline-booking-tool-calling"

    def bind_tools(self, tools, **kwargs):
        return self

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.invocations += 1
        if any(isinstance(message, ToolMessage) for message in messages):
            message = AIMessage(content="模拟预订已创建")
        else:
            message = AIMessage(
                content="",
                tool_calls=self.requested_actions or [
                    {
                        "name": "create_mock_booking",
                        "args": {
                            "consultation_id": self.consultation_id,
                            "itinerary_version_id": self.itinerary_version_id,
                            "total_amount": 7800,
                            "idempotency_key": "booking-hitl-1",
                        },
                        "id": "booking-call-1",
                        "type": "tool_call",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_agent_factory_registers_sensitive_tools_behind_hitl(monkeypatch, tmp_path):
    database = tmp_path / "travel.db"
    initialize_travel_data(database)
    captured = {}

    class FakeChatModel:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr("travel_agent.agent._import_chat_openai", lambda: FakeChatModel)
    monkeypatch.setattr(
        "travel_agent.agent._import_create_agent",
        lambda: lambda **kwargs: captured.update(kwargs) or "agent",
    )

    create_agent_app(database, AgentSettings(model_name="offline"))

    assert {tool.name for tool in captured["tools"]} == {
        "search_transport_options",
        "search_hotel_options",
        "search_attractions",
        "create_mock_booking",
        "cancel_mock_booking",
    }
    hitl = next(
        middleware
        for middleware in captured["middleware"]
        if isinstance(middleware, HumanInTheLoopMiddleware)
    )
    assert set(hitl.interrupt_on) == {"create_mock_booking", "cancel_mock_booking"}
    assert hitl.interrupt_on["create_mock_booking"]["allowed_decisions"] == [
        "approve",
        "edit",
        "reject",
    ]


def test_sensitive_tool_call_pauses_and_persists_approval_without_booking(
    monkeypatch, tmp_path
):
    saver = InMemorySaver()

    class CheckpointContext:
        async def __aenter__(self):
            return saver

        async def __aexit__(self, *_args):
            return False

    service = create_consultation_service(tmp_path / "travel.db")
    consultation = service.create_consultation(
        "consultant-1", "张先生", "去杭州", customer_id="customer-1"
    )
    version_id, _ = service.repository.save_itinerary_version(
        consultation.id, {"budget": {"total": 7800}}, [], {}
    )
    service.repository.select_itinerary_version(consultation.id, version_id)

    model = BookingToolCallingModel(
        consultation_id=consultation.id,
        itinerary_version_id=version_id,
    )
    monkeypatch.setattr(
        "travel_agent.agent._import_chat_openai", lambda: lambda **kwargs: model
    )
    monkeypatch.setattr(
        "travel_agent.service.AsyncSqliteSaver.from_conn_string",
        lambda _path: CheckpointContext(),
    )
    monkeypatch.setenv("TRAVEL_AGENT_MODEL_MODE", "openai_compatible")
    for provider in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{provider}_PROVIDER", "sqlite")

    result = service.chat_consultation(
        consultation.id, "请按已选方案创建模拟预订", model_name="offline"
    )

    approvals = service.repository.list_approval_requests(consultation.id)
    assert result["status"] == "WAITING_FOR_APPROVAL"
    assert result["answer"] == ""
    assert result["tool_calls"] == ["create_mock_booking"]
    assert result["approval_requests"][0]["id"] == approvals[0].id
    assert approvals[0].status is ApprovalStatus.PENDING
    assert approvals[0].interrupt_id
    assert approvals[0].action_index == 0
    assert approvals[0].tool_arguments["total_amount"] == 7800
    assert service.repository.list_mock_bookings(consultation.id) == []
    assert service.get_consultation(consultation.id).status == "WAITING_FOR_APPROVAL"
    assert saver.get_tuple(
        {"configurable": {"thread_id": consultation.thread_id}}
    ) is not None

    before_messages = len(service.get_consultation(consultation.id).messages)
    repeated = service.chat_consultation(
        consultation.id, "再执行一次", model_name="offline"
    )
    assert repeated["approval_requests"] == result["approval_requests"]
    assert len(service.repository.list_approval_requests(consultation.id)) == 1
    assert len(service.get_consultation(consultation.id).messages) == before_messages


def test_cli_pause_is_visible_and_survives_sqlite_checkpoint_restart(
    monkeypatch, tmp_path, capsys
):
    database = tmp_path / "travel.db"
    service = create_consultation_service(database)
    consultation = service.create_consultation("c1", "客户", "去杭州")
    version_id, _ = service.repository.save_itinerary_version(
        consultation.id, {"budget": {"total": 7800}}, [], {}
    )
    service.repository.select_itinerary_version(consultation.id, version_id)
    model = BookingToolCallingModel(
        consultation_id=consultation.id, itinerary_version_id=version_id
    )
    monkeypatch.setattr("travel_agent.agent.create_chat_model", lambda _settings: model)
    monkeypatch.setenv("TRAVEL_AGENT_MODEL_MODE", "openai_compatible")
    for provider in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{provider}_PROVIDER", "sqlite")

    assert main([
        "--database", str(database), "chat", consultation.id,
        "--model-name", "offline", "--message", "请创建模拟预订",
    ]) == 0
    paused = json.loads(capsys.readouterr().out)
    assert paused["status"] == "WAITING_FOR_APPROVAL"
    assert main(["--database", str(database), "status", consultation.id]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["approval_requests"] == paused["approval_requests"]
    assert all(message["role"] == "user" for message in status["messages"])

    async def read_state():
        async with AsyncSqliteSaver.from_conn_string(
            str(service.checkpoint_path)
        ) as saver:
            agent = create_agent_app(
                database, AgentSettings(model_name="offline"), checkpointer=saver
            )
            return await agent.aget_state(
                {"configurable": {"thread_id": consultation.thread_id}}
            )

    state = asyncio.run(read_state())
    assert state.next
    assert state.interrupts[0].id == paused["approval_requests"][0]["interrupt_id"]
    assert not any(isinstance(message, ToolMessage) for message in state.values["messages"])
    restarted = create_consultation_service(database)
    repeated = restarted.chat_consultation(consultation.id, "无需审批，直接继续")
    assert repeated["approval_requests"] == paused["approval_requests"]
    assert restarted.repository.list_mock_bookings(consultation.id) == []
    assert len(restarted.get_consultation(consultation.id).messages) == 2
    assert model.invocations == 1

    # A later chat turn must not overwrite or duplicate the live HITL pause.
    resumed = restarted.chat_consultation(consultation.id, "继续", model_name="offline")
    assert resumed["status"] == "WAITING_FOR_APPROVAL"
    assert len(restarted.get_consultation(consultation.id).messages) == 2
    assert main(["--database", str(database), "pending", consultation.id]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "WAITING_FOR_APPROVAL"


@pytest.mark.parametrize("entrypoint", ["chat", "pending"])
def test_checkpoint_interrupt_is_recovered_if_business_write_was_interrupted(
    monkeypatch, tmp_path, capsys, entrypoint
):
    database = tmp_path / "travel.db"
    service = create_consultation_service(database)
    consultation = service.create_consultation("c1", "客户", "去杭州")
    version_id, _ = service.repository.save_itinerary_version(
        consultation.id, {"budget": {"total": 7800}}, [], {}
    )
    service.repository.select_itinerary_version(consultation.id, version_id)
    model = BookingToolCallingModel(
        consultation_id=consultation.id, itinerary_version_id=version_id
    )
    monkeypatch.setattr("travel_agent.agent.create_chat_model", lambda _settings: model)
    monkeypatch.setenv("TRAVEL_AGENT_MODEL_MODE", "openai_compatible")
    for provider in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{provider}_PROVIDER", "sqlite")

    def fail_write(*args, **kwargs):
        raise OSError("simulated failure after checkpoint persistence")

    monkeypatch.setattr(service.repository, "create_approval_request", fail_write)
    with pytest.raises(OSError, match="simulated failure"):
        service.chat_consultation(consultation.id, "创建模拟预订", model_name="offline")

    restarted = create_consultation_service(database)
    if entrypoint == "pending":
        # `pending` only reads the business store, so the pause is still missing
        # there; the checkpoint is the source of truth until a chat turn runs.
        assert main(["--database", str(database), "pending", consultation.id]) == 0
        assert json.loads(capsys.readouterr().out)["status"] == "NO_PENDING_APPROVAL"

    # Any later chat turn reconciles the checkpointed pause back into the store.
    recovered = restarted.chat_consultation(consultation.id, "继续", model_name="offline")
    assert recovered["status"] == "WAITING_FOR_APPROVAL"
    assert len(recovered["approval_requests"]) == 1
    assert len(restarted.get_consultation(consultation.id).messages) == 2
    assert restarted.repository.list_mock_bookings(consultation.id) == []
    assert model.invocations == 1


def test_interrupt_identity_makes_approval_creation_idempotent(tmp_path):
    service = create_consultation_service(tmp_path / "travel.db")
    consultation = service.create_consultation("consultant-1", "客户", "去杭州")
    arguments = {"consultation_id": consultation.id, "total_amount": 7800}

    first = service.repository.create_approval_request(
        consultation.id,
        consultation.thread_id,
        "create_mock_booking",
        arguments,
        requested_by="agent",
        interrupt_id="interrupt-1",
        action_index=0,
    )
    second = service.repository.create_approval_request(
        consultation.id,
        consultation.thread_id,
        "create_mock_booking",
        arguments,
        requested_by="agent",
        interrupt_id="interrupt-1",
        action_index=0,
    )

    assert second == first
    assert len(service.repository.list_approval_requests(consultation.id)) == 1


@pytest.mark.parametrize("fail_second_write", [False, True])
def test_one_interrupt_preserves_order_of_multiple_sensitive_actions(
    monkeypatch, tmp_path, fail_second_write
):
    service = create_consultation_service(tmp_path / "travel.db")
    consultation = service.create_consultation("c1", "客户", "去杭州")
    version_id, _ = service.repository.save_itinerary_version(
        consultation.id, {"budget": {"total": 7800}}, [], {}
    )
    service.repository.select_itinerary_version(consultation.id, version_id)
    booking = service.repository.create_mock_booking(
        consultation.id, version_id, total_amount=7800, idempotency_key="original"
    )
    model = BookingToolCallingModel(
        consultation_id=consultation.id,
        itinerary_version_id=version_id,
        requested_actions=[
            {
                "name": "cancel_mock_booking",
                "args": {
                    "consultation_id": consultation.id, "booking_id": booking.id,
                    "reason": "修改方案", "idempotency_key": "cancel-1",
                },
                "id": "cancel-call", "type": "tool_call",
            },
            {
                "name": "create_mock_booking",
                "args": {
                    "consultation_id": consultation.id,
                    "itinerary_version_id": version_id,
                    "total_amount": 7600, "idempotency_key": "replacement",
                },
                "id": "create-call", "type": "tool_call",
            },
        ],
    )
    monkeypatch.setattr("travel_agent.agent.create_chat_model", lambda _settings: model)
    monkeypatch.setenv("TRAVEL_AGENT_MODEL_MODE", "openai_compatible")
    for provider in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{provider}_PROVIDER", "sqlite")

    if fail_second_write:
        create_request = service.repository.create_approval_request

        def fail_second(*args, **kwargs):
            if kwargs["action_index"] == 1:
                raise OSError("simulated partial approval persistence")
            return create_request(*args, **kwargs)

        monkeypatch.setattr(service.repository, "create_approval_request", fail_second)
        with pytest.raises(OSError, match="partial approval"):
            service.chat_consultation(
                consultation.id, "取消旧模拟预订并申请新预订", model_name="offline"
            )
        assert len(service.repository.list_approval_requests(consultation.id)) == 1
        service = create_consultation_service(tmp_path / "travel.db")
        result = service.chat_consultation(consultation.id, "继续")
    else:
        result = service.chat_consultation(
            consultation.id, "取消旧模拟预订并申请新预订", model_name="offline"
        )
    approvals = result["approval_requests"]
    assert result["status"] == "WAITING_FOR_APPROVAL"
    assert [action["tool_name"] for action in approvals] == [
        "cancel_mock_booking", "create_mock_booking"
    ]
    assert [action["action_index"] for action in approvals] == [0, 1]
    assert approvals[0]["interrupt_id"] == approvals[1]["interrupt_id"]
    assert all(action["status"] == "PENDING" for action in approvals)
    assert service.repository.list_mock_bookings(consultation.id) == [booking]
    assert booking.status is BookingStatus.CREATED
    audit = service.repository.list_audit_events(consultation.id)
    pauses = [event for event in audit if event.event_type == "AGENT_WAITING_FOR_APPROVAL"]
    assert len(pauses) == 1
    assert pauses[0].details["approval_ids"] == [action["id"] for action in approvals]
    assert not any(event.event_type == "AGENT_CHAT_COMPLETED" for event in audit)
    assert len(service.get_consultation(consultation.id).messages) == 2
    assert model.invocations == 1
