"""Stage 5: human decisions on a paused sensitive action, and the resume path.

The tests here drive the production service boundary (``ConsultationService``)
and the real LangChain agent/callback stack. Only the Chat Model and the
checkpoint storage are substituted, so the assertions describe behaviour a
consultant actually gets from the CLI.
"""

import asyncio
import json
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

import travel_agent.agent as agent_module
import travel_agent.service as service_module
from travel_agent.application import create_consultation_service
from travel_agent.cli import main
from travel_agent.domain import ApprovalStatus, BookingStatus
from travel_agent.provider_factory import ProviderSettings


class BookingToolCallingModel(BaseChatModel):
    """Deterministic model that asks for one sensitive booking action."""

    consultation_id: str = ""
    itinerary_version_id: str = ""
    total_amount: float = 7800
    invocations: int = 0
    tool_results: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "stage5-decision-offline"

    def bind_tools(self, tools, **kwargs):
        return self

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.invocations += 1
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        if tool_messages:
            self.tool_results = [str(m.content) for m in tool_messages]
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="收到审批结果"))]
            )
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "create_mock_booking",
                                "args": {
                                    "consultation_id": self.consultation_id,
                                    "itinerary_version_id": self.itinerary_version_id,
                                    "total_amount": self.total_amount,
                                    "idempotency_key": "decision-key-1",
                                },
                                "id": "create-call-1",
                                "type": "tool_call",
                            }
                        ],
                    )
                )
            ]
        )


class MultiActionModel(BaseChatModel):
    """Deterministic model that asks for two sensitive actions in one turn."""

    consultation_id: str = ""
    itinerary_version_id: str = ""
    booking_id: str = ""
    invocations: int = 0

    @property
    def _llm_type(self) -> str:
        return "stage5-multi-action-offline"

    def bind_tools(self, tools, **kwargs):
        return self

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.invocations += 1
        if any(isinstance(m, ToolMessage) for m in messages):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="改单完成"))]
            )
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "cancel_mock_booking",
                                "args": {
                                    "consultation_id": self.consultation_id,
                                    "booking_id": self.booking_id,
                                    "reason": "修改方案",
                                    "idempotency_key": "cancel-key-1",
                                },
                                "id": "cancel-call-1",
                                "type": "tool_call",
                            },
                            {
                                "name": "create_mock_booking",
                                "args": {
                                    "consultation_id": self.consultation_id,
                                    "itinerary_version_id": self.itinerary_version_id,
                                    "total_amount": 7600,
                                    "idempotency_key": "replacement-key-1",
                                },
                                "id": "create-call-2",
                                "type": "tool_call",
                            },
                        ],
                    )
                )
            ]
        )


@pytest.fixture
def paused(tmp_path, monkeypatch):
    """A consultation paused on a sensitive booking request."""
    saver = InMemorySaver()

    class CheckpointContext:
        async def __aenter__(self):
            return saver

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr(
        service_module.AsyncSqliteSaver, "from_conn_string", lambda _path: CheckpointContext()
    )
    monkeypatch.setenv("TRAVEL_AGENT_MODEL_MODE", "openai_compatible")
    for provider in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{provider}_PROVIDER", "sqlite")

    service = create_consultation_service(tmp_path / "travel.db")
    consultation = service.create_consultation(
        "consultant-1", "张先生", "去杭州", customer_id="customer-1"
    )
    version_id, _ = service.repository.save_itinerary_version(
        consultation.id, {"budget": {"total": 7800}}, [], {}
    )
    service.repository.select_itinerary_version(consultation.id, version_id)
    model = BookingToolCallingModel(
        consultation_id=consultation.id, itinerary_version_id=version_id
    )
    monkeypatch.setattr(agent_module, "_import_chat_openai", lambda: (lambda **kw: model))

    outcome = service.chat_consultation(
        consultation.id, "请按已选方案创建模拟预订", model_name="offline"
    )
    assert outcome["status"] == "WAITING_FOR_APPROVAL"
    approval = service.repository.list_pending_approval_requests(consultation.id)[0]
    return service, consultation, version_id, model, approval


def test_approving_a_paused_action_executes_it_exactly_once(paused):
    service, consultation, version_id, model, approval = paused

    assert service.repository.list_mock_bookings(consultation.id) == []

    result = service.decide_approval(
        approval.id, "APPROVE", decided_by="consultant-1", reason="客户已确认"
    )

    assert result["status"] == "COMPLETED"
    assert result["answer"] == "收到审批结果"
    bookings = service.repository.list_mock_bookings(consultation.id)
    assert len(bookings) == 1
    assert bookings[0].status is BookingStatus.CREATED
    assert bookings[0].total_amount == 7800
    assert bookings[0].itinerary_version_id == version_id
    assert service.get_consultation(consultation.id).status == "OPEN"
    assert service.list_pending_approvals(consultation.id) == []

    decided = service.repository.get_approval_request(approval.id)
    assert decided.status is ApprovalStatus.APPROVED
    assert decided.decided_by == "consultant-1"
    assert decided.decision_reason == "客户已确认"


def test_rejection_does_not_execute_and_lets_the_agent_replan(paused):
    service, consultation, _version_id, model, approval = paused

    result = service.reject_approval(
        approval.id, decided_by="consultant-1", reason="客户暂不预订"
    )

    assert result["status"] == "COMPLETED"
    assert service.repository.list_mock_bookings(consultation.id) == []
    # The middleware hands the model an error ToolMessage carrying the reason.
    assert any("客户暂不预订" in content for content in model.tool_results)
    rejected = service.repository.get_approval_request(approval.id)
    assert rejected.status is ApprovalStatus.REJECTED
    assert rejected.decision_reason == "客户暂不预订"
    assert service.get_consultation(consultation.id).status == "OPEN"


def test_editing_an_action_executes_the_edited_arguments(paused):
    service, consultation, _version_id, _model, approval = paused

    result = service.decide_approval(
        approval.id,
        "EDIT",
        decided_by="consultant-1",
        edited_arguments={**approval.tool_arguments, "total_amount": 6500},
    )

    assert result["status"] == "COMPLETED"
    bookings = service.repository.list_mock_bookings(consultation.id)
    assert [booking.total_amount for booking in bookings] == [6500]
    decided = service.repository.get_approval_request(approval.id)
    assert decided.status is ApprovalStatus.APPROVED
    assert decided.edited_arguments["total_amount"] == 6500


def test_a_decided_pause_cannot_be_decided_twice(paused):
    service, consultation, _version_id, _model, approval = paused

    service.decide_approval(approval.id, "APPROVE", decided_by="consultant-1")

    with pytest.raises(ValueError, match="already decided"):
        service.decide_approval(approval.id, "REJECT", decided_by="consultant-2", reason="反悔")
    with pytest.raises(ValueError, match="already decided"):
        service.reject_approval(approval.id, decided_by="consultant-2", reason="反悔")

    assert len(service.repository.list_mock_bookings(consultation.id)) == 1


def test_resuming_an_already_executed_pause_is_a_no_op(paused):
    service, consultation, _version_id, model, approval = paused
    service.decide_approval(approval.id, "APPROVE", decided_by="consultant-1")
    invocations = model.invocations

    again = service.resume_approved_consultation(consultation.id)

    assert again["status"] == "APPROVAL_COMPLETED"
    assert len(service.repository.list_mock_bookings(consultation.id)) == 1
    assert model.invocations == invocations


def test_editing_cannot_change_the_tool_name(paused):
    service, _consultation, _version_id, _model, approval = paused

    with pytest.raises(ValueError, match="tool name"):
        service.decide_approval(
            approval.id,
            "EDIT",
            decided_by="consultant-1",
            edited_arguments={"tool_name": "cancel_mock_booking"},
        )


def test_editing_cannot_inject_unknown_or_invalid_arguments(paused):
    service, consultation, _version_id, _model, approval = paused

    with pytest.raises(ValueError, match="only override existing arguments"):
        service.decide_approval(
            approval.id,
            "EDIT",
            decided_by="consultant-1",
            edited_arguments={"smuggled_field": "value"},
        )
    with pytest.raises(ValueError, match="rejected by tool schema"):
        service.decide_approval(
            approval.id,
            "EDIT",
            decided_by="consultant-1",
            edited_arguments={"total_amount": -1},
        )

    # A rejected edit must leave the pause untouched and executable later.
    still_pending = service.repository.get_approval_request(approval.id)
    assert still_pending.status is ApprovalStatus.PENDING
    assert service.repository.list_mock_bookings(consultation.id) == []

    approved = service.decide_approval(
        approval.id, "APPROVE", decided_by="consultant-1"
    )
    assert approved["status"] == "COMPLETED"
    assert [booking.total_amount for booking in service.repository.list_mock_bookings(consultation.id)] == [7800]


def test_an_edited_action_keeps_every_required_argument(paused):
    service, consultation, version_id, _model, approval = paused

    # The reviewer only changes the amount; the identity fields must survive the
    # wholesale argument replacement performed by HumanInTheLoopMiddleware.
    service.decide_approval(
        approval.id,
        "EDIT",
        decided_by="consultant-1",
        edited_arguments={"total_amount": 6500},
    )

    bookings = service.repository.list_mock_bookings(consultation.id)
    assert len(bookings) == 1
    assert bookings[0].consultation_id == consultation.id
    assert bookings[0].itinerary_version_id == version_id
    assert bookings[0].idempotency_key == "decision-key-1"
    assert bookings[0].total_amount == 6500


def test_decision_without_resume_keeps_the_pause_open(paused):
    service, consultation, _version_id, _model, approval = paused

    view = service.decide_approval(
        approval.id, "APPROVE", decided_by="consultant-1", resume=False
    )

    assert view["status"] == "APPROVAL_COMPLETED"
    assert service.repository.list_mock_bookings(consultation.id) == []

    resumed = service.resume_approved_consultation(consultation.id)
    assert resumed["status"] == "COMPLETED"
    assert len(service.repository.list_mock_bookings(consultation.id)) == 1


def test_every_action_in_one_pause_is_decided_before_execution(tmp_path, monkeypatch):
    saver = InMemorySaver()

    class CheckpointContext:
        async def __aenter__(self):
            return saver

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr(
        service_module.AsyncSqliteSaver, "from_conn_string", lambda _path: CheckpointContext()
    )
    monkeypatch.setenv("TRAVEL_AGENT_MODEL_MODE", "openai_compatible")
    for provider in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{provider}_PROVIDER", "sqlite")

    service = create_consultation_service(tmp_path / "travel.db")
    consultation = service.create_consultation("c1", "客户", "去杭州")
    version_id, _ = service.repository.save_itinerary_version(
        consultation.id, {"budget": {"total": 7800}}, [], {}
    )
    service.repository.select_itinerary_version(consultation.id, version_id)
    booking = service.repository.create_mock_booking(
        consultation.id, version_id, total_amount=7800, idempotency_key="original-key"
    )
    model = MultiActionModel(
        consultation_id=consultation.id,
        itinerary_version_id=version_id,
        booking_id=booking.id,
    )
    monkeypatch.setattr(agent_module, "_import_chat_openai", lambda: (lambda **kw: model))

    pause = service.chat_consultation(
        consultation.id, "取消旧预订并申请新预订", model_name="offline"
    )
    actions = pause["approval_requests"]
    assert [action["tool_name"] for action in actions] == [
        "cancel_mock_booking",
        "create_mock_booking",
    ]
    assert [action["action_index"] for action in actions] == [0, 1]
    assert len({action["interrupt_id"] for action in actions}) == 1

    partial = service.decide_approval(
        actions[0]["id"], "APPROVE", decided_by="consultant-1"
    )
    assert partial["status"] == "WAITING_FOR_APPROVAL"
    assert [item.status for item in service.repository.list_mock_bookings(consultation.id)] == [
        BookingStatus.CREATED
    ]

    completed = service.decide_approval(
        actions[1]["id"], "APPROVE", decided_by="consultant-1"
    )
    assert completed["status"] == "COMPLETED"
    final = service.repository.list_mock_bookings(consultation.id)
    assert [(item.id, item.status, item.total_amount) for item in final] == [
        (booking.id, BookingStatus.CANCELLED, 7800),
        (final[1].id, BookingStatus.CREATED, 7600),
    ]
    assert model.invocations == 2


def test_cli_approve_and_reject_drive_the_pause(tmp_path, monkeypatch, capsys):
    saver = InMemorySaver()

    class CheckpointContext:
        async def __aenter__(self):
            return saver

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr(
        service_module.AsyncSqliteSaver, "from_conn_string", lambda _path: CheckpointContext()
    )
    monkeypatch.setenv("TRAVEL_AGENT_MODEL_MODE", "openai_compatible")
    for provider in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{provider}_PROVIDER", "sqlite")

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
    monkeypatch.setattr(agent_module, "create_chat_model", lambda _settings: model)

    assert main([
        "--database", str(database), "chat", consultation.id,
        "--model-name", "offline", "--message", "创建预订",
    ]) == 0
    paused = json.loads(capsys.readouterr().out)
    approval_id = paused["approval_requests"][0]["id"]

    assert main([
        "--database", str(database), "pending", consultation.id,
    ]) == 0
    pending = json.loads(capsys.readouterr().out)
    assert pending["status"] == "WAITING_FOR_APPROVAL"
    assert [item["id"] for item in pending["approval_requests"]] == [approval_id]

    # A rejection must state a reason, and the CLI reports it as an argument error.
    with pytest.raises(SystemExit) as error:
        main([
            "--database", str(database), "reject", approval_id,
            "--decided-by", "consultant-1",
        ])
    assert error.value.code == 2

    assert main([
        "--database", str(database), "approve", approval_id,
        "--decided-by", "consultant-1", "--reason", "客户已确认",
        "--edit-json", json.dumps({"total_amount": 6500}),
    ]) == 0
    approved = json.loads(capsys.readouterr().out)
    assert approved["status"] == "COMPLETED"
    assert approved["answer"] == "收到审批结果"
    bookings = service.repository.list_mock_bookings(consultation.id)
    assert [booking.total_amount for booking in bookings] == [6500]

    assert main([
        "--database", str(database), "status", consultation.id,
    ]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "OPEN"
    assert status["approval_requests"] == []


def test_a_crash_between_deciding_and_resuming_stays_recoverable(paused):
    service, consultation, _version_id, _model, approval = paused

    # Record the decision, then simulate the process dying before the graph ran.
    service.decide_approval(
        approval.id, "APPROVE", decided_by="consultant-1", resume=False
    )
    assert service.repository.list_mock_bookings(consultation.id) == []

    # A brand new service (as a restarted CLI would build) still completes it once.
    restarted = create_consultation_service(service.repository.database_path)
    recovered = restarted.resume_approved_consultation(consultation.id)
    assert recovered["status"] == "COMPLETED"
    assert len(restarted.repository.list_mock_bookings(consultation.id)) == 1

    # Repeating the recovery must not create a second booking.
    repeated = restarted.resume_approved_consultation(consultation.id)
    assert repeated["status"] == "APPROVAL_COMPLETED"
    assert len(restarted.repository.list_mock_bookings(consultation.id)) == 1


def test_pending_approval_blocks_new_chat_turns(paused):
    service, consultation, _version_id, model, _approval = paused
    invocations = model.invocations
    messages_before = len(service.get_consultation(consultation.id).messages)

    blocked = service.chat_consultation(consultation.id, "顺便帮我查个酒店", model_name="offline")

    assert blocked["status"] == "WAITING_FOR_APPROVAL"
    assert blocked["answer"] == ""
    assert model.invocations == invocations
    assert len(service.get_consultation(consultation.id).messages) == messages_before
