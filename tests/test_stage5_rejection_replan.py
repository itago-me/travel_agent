"""Stage 5: rejecting a sensitive action and replanning towards a new request.

This is the acceptance criterion "拒绝后任务可以继续修改方案" expressed as a test:
a refusal must not end the task, and the Agent's follow-up request must be able
to pause again, be approved, and execute exactly once.
"""

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

import travel_agent.agent as agent_module
import travel_agent.service as service_module
from travel_agent.application import create_consultation_service
from travel_agent.domain import ApprovalStatus, BookingStatus


class ReplanningModel(BaseChatModel):
    """Asks to book at 7800; after a refusal asks again at 6000."""

    consultation_id: str = ""
    itinerary_version_id: str = ""
    invocations: int = 0
    tool_results: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "stage5-replan-offline"

    def bind_tools(self, tools, **kwargs):
        return self

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.invocations += 1
        self.tool_results = [
            str(message.content)
            for message in messages
            if isinstance(message, ToolMessage)
        ]
        if any("booking_id" in content for content in self.tool_results):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="新方案预订已提交"))]
            )
        if any("rejected" in content for content in self.tool_results):
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
                                        "total_amount": 6000,
                                        "idempotency_key": "replan-key-1",
                                    },
                                    "id": "replan-call",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    )
                ]
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
                                    "total_amount": 7800,
                                    "idempotency_key": "first-key-1",
                                },
                                "id": "first-call",
                                "type": "tool_call",
                            }
                        ],
                    )
                )
            ]
        )


@pytest.fixture
def replanning(tmp_path, monkeypatch):
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
    consultation = service.create_consultation("consultant-1", "张先生", "去杭州")
    version_id, _ = service.repository.save_itinerary_version(
        consultation.id, {"budget": {"total": 7800}}, [], {}
    )
    service.repository.select_itinerary_version(consultation.id, version_id)
    model = ReplanningModel(
        consultation_id=consultation.id, itinerary_version_id=version_id
    )
    monkeypatch.setattr(agent_module, "_import_chat_openai", lambda: (lambda **kw: model))

    first_pause = service.chat_consultation(
        consultation.id, "请按已选方案创建模拟预订", model_name="offline"
    )
    assert first_pause["status"] == "WAITING_FOR_APPROVAL"
    return service, consultation, version_id, model, first_pause["approval_requests"][0]["id"]


def test_rejection_lets_the_agent_replan_and_pause_again(replanning):
    service, consultation, _version_id, model, first_approval_id = replanning
    assert service.repository.list_approval_requests(consultation.id)[0].status is (
        ApprovalStatus.PENDING
    )

    replanned = service.reject_approval(
        first_approval_id, decided_by="consultant-1", reason="预算超了"
    )

    # The refusal does not end the task: the Agent immediately proposes a cheap
    # alternative, which pauses again instead of executing.
    assert replanned["status"] == "WAITING_FOR_APPROVAL"
    assert replanned["answer"] == ""
    pending = replanned["approval_requests"]
    assert [action["tool_name"] for action in pending] == ["create_mock_booking"]
    assert pending[0]["tool_arguments"]["total_amount"] == 6000
    assert pending[0]["id"] != first_approval_id
    assert service.repository.list_mock_bookings(consultation.id) == []

    # The refused action stays refused, and the new one is what is now pending.
    first = service.repository.get_approval_request(first_approval_id)
    assert first.status is ApprovalStatus.REJECTED
    assert service.repository.get_approval_request(pending[0]["id"]).status is (
        ApprovalStatus.PENDING
    )
    assert any("rejected" in content for content in model.tool_results)


def test_approving_the_replanned_action_creates_one_booking(replanning):
    service, consultation, version_id, _model, first_approval_id = replanning
    service.reject_approval(
        first_approval_id, decided_by="consultant-1", reason="预算超了"
    )
    second = service.list_pending_approvals(consultation.id)[0]

    completed = service.decide_approval(
        second.id, "APPROVE", decided_by="consultant-1", reason="这个方案可以"
    )

    assert completed["status"] == "COMPLETED"
    assert completed["answer"] == "新方案预订已提交"
    bookings = service.repository.list_mock_bookings(consultation.id)
    assert [(booking.total_amount, booking.status) for booking in bookings] == [
        (6000, BookingStatus.CREATED)
    ]
    assert bookings[0].itinerary_version_id == version_id
    assert service.get_consultation(consultation.id).status == "OPEN"
    assert service.list_pending_approvals(consultation.id) == []

    # Both pauses are closed out, so neither can shadow a later one.
    audit = service.repository.list_audit_events(consultation.id)
    resumed = [event for event in audit if event.event_type == "APPROVAL_RESUMED"]
    assert len(resumed) == 2
    assert {event.details["interrupt_id"] for event in resumed} == {
        service.repository.get_approval_request(first_approval_id).interrupt_id,
        second.interrupt_id,
    }
    assert sum(1 for event in audit if event.event_type == "AGENT_CHAT_COMPLETED") == 1


def test_a_second_replan_can_still_be_approved(replanning):
    """A consumed pause must not block the next one (regression guard)."""
    service, consultation, _version_id, _model, first_approval_id = replanning
    service.reject_approval(
        first_approval_id, decided_by="consultant-1", reason="预算超了"
    )
    second = service.list_pending_approvals(consultation.id)[0]

    # Rejecting the replanned offer too leaves the task open once more.
    third_pause = service.reject_approval(
        second.id, decided_by="consultant-1", reason="还是不合适"
    )
    assert third_pause["status"] == "WAITING_FOR_APPROVAL"
    third = service.list_pending_approvals(consultation.id)[0]
    assert third.id not in {first_approval_id, second.id}

    completed = service.decide_approval(third.id, "APPROVE", decided_by="consultant-1")
    assert completed["status"] == "COMPLETED"
    assert len(service.repository.list_mock_bookings(consultation.id)) == 1
    assert len(service.repository.list_approval_requests(consultation.id)) == 3
