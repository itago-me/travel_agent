import asyncio
import json
from datetime import date
from typing import Any
from uuid import uuid4
from pathlib import Path

from .domain import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    AuditEvent,
    Consultation,
    CustomerPreference,
    Message,
)
from .repository import SQLiteRepository
from .requirement_extractor import extract_requirements
from .agent import AgentSettings, create_agent_app, load_agent_settings
from .langchain_tools import create_sensitive_booking_tools, create_travel_tools
from .provider_factory import ProviderSettings
from langchain_core.messages import RemoveMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command
from pydantic import ValidationError

from .booking import CancelMockBookingRequest, CreateMockBookingRequest


# The graph is only allowed to stop on these two high-risk actions. The set is
# duplicated from the middleware configuration on purpose: the service must not
# resume a pause it does not understand.
HITL_TOOL_NAMES = ("create_mock_booking", "cancel_mock_booking")


def _last_message_content(messages: list[Any]) -> str:
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if content:
            return str(content)
        if isinstance(message, dict) and message.get("content"):
            return str(message["content"])
    return ""


def _message_as_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    return {
        "type": getattr(message, "type", None),
        "content": getattr(message, "content", ""),
        "tool_calls": getattr(message, "tool_calls", []),
    }


def _tool_history_status(messages: list[Any]) -> str:
    """Classify OpenAI tool-call ordering as complete, pending, or invalid."""
    pending: set[str] = set()
    for message in messages:
        message_type = getattr(message, "type", None)
        if pending:
            if message_type != "tool":
                return "invalid"
            tool_call_id = getattr(message, "tool_call_id", None)
            if tool_call_id not in pending:
                return "invalid"
            pending.remove(tool_call_id)
            continue
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            pending = {call["id"] for call in tool_calls if call.get("id")}
    return "pending" if pending else "complete"


def _loads_message_content(content: Any) -> dict[str, Any] | None:
    """Decode a ToolMessage body into the dict the tool actually returned."""
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _option_list(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    options = payload.get("options")
    if not isinstance(options, list):
        return []
    return [option for option in options if isinstance(option, dict)]


def _first_option(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    options = _option_list(payload)
    return options[0] if options else None


def _estimate_budget(
    transport: dict[str, Any],
    hotel: dict[str, Any],
    attractions: list[dict[str, Any]],
    requirements: dict[str, Any],
    traveler_count: int,
) -> dict[str, Any] | None:
    """Recompute the trip total from provider output.

    Kept deliberately strict: it returns nothing rather than guessing when the
    dates or prices it needs are missing, so a version never carries a fabricated
    total.
    """
    try:
        check_in = date.fromisoformat(str(requirements["start_date"]))
        check_out = date.fromisoformat(str(requirements["end_date"]))
    except (KeyError, TypeError, ValueError):
        return None
    nights = (check_out - check_in).days
    if nights <= 0:
        return None

    transport_total = transport.get("total_price")
    price_per_night = hotel.get("price_per_night")
    if transport_total is None or price_per_night is None:
        return None
    transport_total = float(transport_total)
    hotel_total = float(price_per_night) * nights
    # Ticket prices are per traveler, so they scale with the party size.
    attractions_total = (
        sum(float(item.get("ticket_price") or 0) for item in attractions)
        * traveler_count
    )
    total = transport_total + hotel_total + attractions_total
    budget_limit = requirements.get("budget")
    return {
        "transport_total": transport_total,
        "hotel_total": hotel_total,
        "attractions_total": attractions_total,
        "total": total,
        "budget_limit": float(budget_limit) if budget_limit is not None else None,
        "within_budget": budget_limit is not None and total <= float(budget_limit),
        "nights": nights,
        "traveler_count": traveler_count,
    }


_VERSION_TOOLS = (
    "search_transport_options",
    "search_hotel_options",
    "search_attractions",
)


class ConsultationService:
    def __init__(
        self,
        repository: SQLiteRepository,
        checkpoint_path: str | Path | None = None,
    ):
        self.repository = repository
        self.checkpoint_path = Path(
            checkpoint_path
            or repository.database_path.with_name(
                f"{repository.database_path.stem}_checkpoints.sqlite"
            )
        )

    def create_consultation(
        self,
        consultant_id: str,
        customer_name: str,
        initial_message: str,
        *,
        customer_id: str | None = None,
    ) -> Consultation:
        customer_id = customer_id or f"legacy:{customer_name}"
        consultation = Consultation(
            id=str(uuid4()),
            thread_id=str(uuid4()),
            consultant_id=consultant_id,
            customer_name=customer_name,
            status="OPEN",
            customer_id=customer_id,
            messages=[Message("user", initial_message)],
        )
        self.repository.create_consultation(consultation)
        self.repository.add_audit_event(
            AuditEvent(
                consultation_id=consultation.id,
                event_type="CONSULTATION_CREATED",
                details={
                    "consultant_id": consultant_id,
                    "customer_id": consultation.customer_id,
                    "thread_id": consultation.thread_id,
                },
            )
        )
        # The opening message usually already carries dates, party size and
        # budget. Extracting them here means every later turn (and the itinerary
        # version built from its search results) has the requirements it needs.
        self._absorb_requirements(consultation.id, initial_message)
        return consultation

    def _absorb_requirements(
        self, consultation_id: str, message: str
    ) -> dict[str, Any]:
        """Merge whatever a user message states explicitly into requirements."""
        extracted = extract_requirements(message)
        if not extracted:
            return {}
        consultation = self.get_consultation(consultation_id)
        merged = {**consultation.trip_requirements, **extracted}
        self.repository.update_requirements(consultation_id, merged)
        self.repository.add_audit_event(
            AuditEvent(
                consultation_id=consultation_id,
                event_type="REQUIREMENTS_EXTRACTED",
                details={"fields": sorted(extracted), "source": "user_message"},
            )
        )
        return extracted

    def get_consultation(self, consultation_id: str) -> Consultation:
        return self.repository.get_consultation(consultation_id)

    def set_customer_preference(
        self,
        customer_id: str,
        preference_key: str,
        value: Any,
        *,
        source: str = "explicit",
    ) -> CustomerPreference:
        return self.repository.set_customer_preference(
            customer_id,
            preference_key,
            value,
            source=source,
        )

    def list_customer_preferences(self, customer_id: str) -> list[CustomerPreference]:
        return self.repository.list_customer_preferences(customer_id)

    def delete_customer_preference(self, customer_id: str, preference_key: str) -> bool:
        return self.repository.delete_customer_preference(customer_id, preference_key)

    def get_pending_approval_result(
        self, consultation_id: str
    ) -> dict[str, Any] | None:
        consultation = self.get_consultation(consultation_id)
        if self.checkpoint_path.exists():

            async def read_interrupts():
                async with AsyncSqliteSaver.from_conn_string(
                    str(self.checkpoint_path)
                ) as checkpointer:
                    checkpoint = await checkpointer.aget_tuple(
                        {"configurable": {"thread_id": consultation.thread_id}}
                    )
                    return [
                        graph_interrupt
                        for _, channel, values in (
                            checkpoint.pending_writes if checkpoint else []
                        )
                        # Persisted interrupt channel used by the SQLite saver.
                        if channel == "__interrupt__"
                        for graph_interrupt in values
                    ]

            # Inspect persisted interrupts without constructing or calling a model.
            # Always reconcile the whole batch, even if some approvals already exist.
            interrupts = asyncio.run(read_interrupts())
            if interrupts:
                approvals = self._record_approval_interrupts(
                    consultation, {"__interrupt__": interrupts}
                )
                return self._waiting_for_approval_result(
                    consultation, approvals, messages=[]
                )
        approvals = self.repository.list_pending_approval_requests(consultation_id)
        if not approvals:
            return None
        if consultation.status != "WAITING_FOR_APPROVAL":
            self.repository.update_consultation_status(
                consultation_id, "WAITING_FOR_APPROVAL"
            )
        return self._waiting_for_approval_result(consultation, approvals, messages=[])

    def _itinerary_version_state(self, consultation_id: str) -> dict[str, Any]:
        """Current version list for the model.

        Refreshed every turn because each chat turn can append a version; the
        model must see the newest ids or it will reuse a stale one when calling
        ``create_mock_booking``.
        """
        versions = self.repository.list_itinerary_versions(consultation_id)
        selected = self.repository.get_selected_itinerary(consultation_id)
        return {
            "itinerary_versions": [
                {
                    "id": version["id"],
                    "version_number": version["version_number"],
                    "payload": version["payload"],
                }
                for version in versions
            ],
            "selected_itinerary_version_id": (
                selected["version_id"] if selected else None
            ),
        }

    def _build_agent_tools(self, provider_settings: ProviderSettings) -> list[Any]:
        """Build the exact tool set the production Agent is allowed to bind.

        Query tools come from the configured provider bundle; the two sensitive
        booking tools are always appended because ``HumanInTheLoopMiddleware``
        refuses to let them execute without a human decision.
        """
        return [
            *create_travel_tools(
                self.repository.database_path,
                provider_settings=provider_settings,
            ),
            *create_sensitive_booking_tools(self.repository.database_path),
        ]

    def chat_consultation(
        self,
        consultation_id: str,
        user_message: str,
        *,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        env_file: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run one real model/tool turn for a consultation."""
        consultation = self.get_consultation(consultation_id)
        pending_result = self.get_pending_approval_result(consultation_id)
        if pending_result is not None:
            return pending_result
        loaded = load_agent_settings(env_file)
        settings = AgentSettings(
            model_mode=loaded.model_mode,
            model_name=model_name or loaded.model_name,
            api_key=api_key or loaded.api_key,
            base_url=base_url or loaded.base_url,
        )
        provider_settings = ProviderSettings.from_env()
        tools = self._build_agent_tools(provider_settings)

        async def run_chat():
            async with AsyncSqliteSaver.from_conn_string(
                str(self.checkpoint_path)
            ) as checkpointer:
                config = {"configurable": {"thread_id": consultation.thread_id}}
                checkpoint = await checkpointer.aget_tuple(config)
                # Read once per new thread, before constructing the Agent.
                # Existing checkpoints predating 4.4 get their snapshot on next chat.
                saved_values = getattr(checkpoint, "checkpoint", {}).get(
                    "channel_values", {}
                )
                needs_snapshot = checkpoint is None or (
                    hasattr(checkpoint, "checkpoint")
                    and "customer_preferences" not in saved_values
                )
                preference_state = {}
                if needs_snapshot:
                    preference_state = {
                        "customer_id": consultation.customer_id,
                        "customer_preferences": {
                            preference.preference_key: preference.value
                            for preference in self.list_customer_preferences(
                                consultation.customer_id
                            )
                        },
                    }
                agent = create_agent_app(
                    self.repository.database_path,
                    settings,
                    checkpointer=checkpointer,
                    tools=tools,
                )
                history_status = "complete"
                if checkpoint is not None:
                    state = await agent.aget_state(config)
                    interrupts = getattr(state, "interrupts", ())
                    if interrupts:
                        # A process may have stopped between checkpoint and business
                        # persistence. Reconcile that pause before accepting new input.
                        return {**state.values, "__interrupt__": interrupts}
                    history_status = _tool_history_status(
                        list(state.values.get("messages", []))
                    )
                    if history_status == "pending" and state.next:
                        continued = await agent.ainvoke(None, config=config)
                        if continued.get("__interrupt__"):
                            return continued

                self.append_message(consultation_id, "user", user_message)
                current_requirements = self._absorb_requirements(
                    consultation_id, user_message
                )
                trip_requirements = dict(
                    self.get_consultation(consultation_id).trip_requirements
                )
                turn_state = {
                    "current_trip_requirements": current_requirements,
                    **preference_state,
                    **self._itinerary_version_state(consultation_id),
                }
                if trip_requirements:
                    turn_state["trip_requirements"] = trip_requirements
                if checkpoint is None:
                    input_messages = [
                        {"role": message.role, "content": message.content}
                        for message in self.get_consultation(consultation_id).messages
                    ]
                    return await agent.ainvoke(
                        {"messages": input_messages, **turn_state}, config=config
                    )

                if history_status == "invalid":
                    business_messages = [
                        {"role": message.role, "content": message.content}
                        for message in self.get_consultation(consultation_id).messages
                    ]
                    await agent.aupdate_state(
                        config,
                        {
                            **turn_state,
                            "messages": [
                                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                                *business_messages,
                            ],
                        },
                        as_node="__start__",
                    )
                    return await agent.ainvoke(None, config=config)

                next_input = {"messages": [{"role": "user", "content": user_message}]}
                if history_status != "pending" or current_requirements:
                    next_input.update(turn_state)
                # 版本列表每轮都要重发：本轮 chat 会追加新版本，只在上面那个
                # 分支下发会让模型在后续轮次里仍看到旧列表，从而挑到过期 id。
                next_input.update(self._itinerary_version_state(consultation_id))
                return await agent.ainvoke(next_input, config=config)

        result = asyncio.run(run_chat())
        return self._finalize_agent_result(consultation, result)

    def _record_approval_interrupts(
        self, consultation: Consultation, result: dict[str, Any]
    ) -> list[ApprovalRequest]:
        approvals = []
        for graph_interrupt in result.get("__interrupt__", ()):
            interrupt_id = getattr(graph_interrupt, "id", None)
            payload = getattr(graph_interrupt, "value", None)
            if not interrupt_id or not isinstance(payload, dict):
                continue
            for action_index, action in enumerate(payload.get("action_requests", [])):
                if not isinstance(action, dict):
                    continue
                tool_name = action.get("name")
                tool_arguments = action.get("args")
                if not isinstance(tool_name, str) or not isinstance(
                    tool_arguments, dict
                ):
                    continue
                approvals.append(
                    self.repository.create_approval_request(
                        consultation.id,
                        consultation.thread_id,
                        tool_name,
                        tool_arguments,
                        requested_by="agent",
                        interrupt_id=interrupt_id,
                        action_index=action_index,
                    )
                )
        if approvals and consultation.status != "WAITING_FOR_APPROVAL":
            self.repository.update_consultation_status(
                consultation.id, "WAITING_FOR_APPROVAL"
            )
            self.repository.add_audit_event(
                AuditEvent(
                    consultation_id=consultation.id,
                    event_type="AGENT_WAITING_FOR_APPROVAL",
                    details={
                        "thread_id": consultation.thread_id,
                        "approval_ids": [approval.id for approval in approvals],
                        "tool_calls": [approval.tool_name for approval in approvals],
                    },
                )
            )
        return approvals

    @staticmethod
    def _waiting_for_approval_result(
        consultation: Consultation,
        approvals: list[ApprovalRequest],
        *,
        messages: list[Any],
    ) -> dict[str, Any]:
        return {
            "consultation_id": consultation.id,
            "thread_id": consultation.thread_id,
            "status": "WAITING_FOR_APPROVAL",
            "answer": "",
            "tool_calls": [approval.tool_name for approval in approvals],
            "approval_requests": [approval.as_dict() for approval in approvals],
            "messages": [_message_as_dict(message) for message in messages],
        }

    # ------------------------------------------------------------------
    # Human-in-the-loop decisions
    # ------------------------------------------------------------------

    def list_pending_approvals(self, consultation_id: str) -> list[ApprovalRequest]:
        """Return the pending approvals a consultant still has to decide.

        The consultation is loaded first so an unknown id fails loudly instead of
        looking like a consultation with nothing to approve.
        """
        self.get_consultation(consultation_id)
        return self.repository.list_pending_approval_requests(consultation_id)

    def decide_approval(
        self,
        approval_id: str,
        decision: ApprovalDecision | str,
        *,
        decided_by: str,
        reason: str | None = None,
        edited_arguments: dict[str, Any] | None = None,
        resume: bool = True,
    ) -> dict[str, Any]:
        """Record one human decision and, when the pause is complete, continue it.

        A single interrupt may contain several actions (for example "cancel the old
        booking, then create a new one"). ``HumanInTheLoopMiddleware`` refuses to
        resume unless every action carries exactly one decision, so this method
        only continues the graph once no PENDING approval remains for the pause.
        """
        approval = self.repository.get_approval_request(approval_id)
        decision = ApprovalDecision(decision)
        self._validate_decision(approval, decision, edited_arguments)
        merged_arguments = (
            self._merge_edited_arguments(approval, edited_arguments)
            if decision is ApprovalDecision.EDIT and edited_arguments
            else None
        )
        decided = self.repository.decide_approval_request(
            approval_id,
            decision,
            decided_by=decided_by,
            reason=reason,
            edited_arguments=merged_arguments,
        )
        self.repository.add_audit_event(
            AuditEvent(
                consultation_id=approval.consultation_id,
                event_type="APPROVAL_DECIDED",
                details={
                    "thread_id": approval.thread_id,
                    "approval_id": decided.id,
                    "interrupt_id": decided.interrupt_id,
                    "action_index": decided.action_index,
                    "tool_name": decided.tool_name,
                    "decision": decision.value,
                    "decided_by": decided_by,
                    "reason": reason,
                    "edited_arguments": merged_arguments,
                },
            )
        )
        if not resume:
            return self._approval_progress_result(approval.consultation_id, decided)
        return self.resume_approved_consultation(approval.consultation_id)

    def reject_approval(
        self,
        approval_id: str,
        *,
        decided_by: str,
        reason: str,
        resume: bool = True,
    ) -> dict[str, Any]:
        """Reject a sensitive action; the Agent keeps running and may replan."""
        return self.decide_approval(
            approval_id,
            ApprovalDecision.REJECT,
            decided_by=decided_by,
            reason=reason,
            resume=resume,
        )

    @staticmethod
    def _validate_decision(
        approval: ApprovalRequest,
        decision: ApprovalDecision,
        edited_arguments: dict[str, Any] | None,
    ) -> None:
        if approval.status is not ApprovalStatus.PENDING:
            raise ValueError(
                f"Approval request already decided: {approval.status.value}"
            )
        if approval.tool_name not in HITL_TOOL_NAMES:
            raise ValueError(
                f"Approval request is not a HITL action: {approval.tool_name}"
            )
        if decision is ApprovalDecision.EDIT:
            if not edited_arguments:
                raise ValueError("edited_arguments must not be empty for EDIT")
            # Editing must not turn a booking action into another tool call.
            if "tool_name" in edited_arguments or "name" in edited_arguments:
                raise ValueError("edited_arguments must not change the tool name")
            unknown = set(edited_arguments) - set(approval.tool_arguments)
            if unknown:
                raise ValueError(
                    "edited_arguments must only override existing arguments: "
                    + ", ".join(sorted(unknown))
                )

    def _merge_edited_arguments(
        self, approval: ApprovalRequest, edited_arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply a reviewer's edit as a patch over the Agent's original arguments.

        ``HumanInTheLoopMiddleware`` replaces the tool arguments with the edited
        action wholesale, so a partial edit would drop required fields such as the
        consultation or itinerary id. The merged result is validated against the
        tool schema here, before it can reach the graph.
        """
        merged = {**approval.tool_arguments, **edited_arguments}
        schemas = {
            "create_mock_booking": CreateMockBookingRequest,
            "cancel_mock_booking": CancelMockBookingRequest,
        }
        model = schemas.get(approval.tool_name)
        if model is None:
            raise ValueError(
                f"Approval request is not a HITL action: {approval.tool_name}"
            )
        try:
            model.model_validate(merged)
        except ValidationError as error:
            raise ValueError(
                f"edited_arguments rejected by tool schema: {error}"
            ) from error
        return merged

    def _approval_progress_result(
        self, consultation_id: str, decided: ApprovalRequest
    ) -> dict[str, Any]:
        consultation = self.get_consultation(consultation_id)
        pending = self.list_pending_approvals(consultation_id)
        return {
            "consultation_id": consultation_id,
            "thread_id": consultation.thread_id,
            "status": ("WAITING_FOR_APPROVAL" if pending else "APPROVAL_COMPLETED"),
            "decision": decided.as_dict(),
            "approval_requests": [approval.as_dict() for approval in pending],
        }

    def resume_approved_consultation(self, consultation_id: str) -> dict[str, Any]:
        """Continue an interrupted thread once every action has a decision."""
        consultation = self.get_consultation(consultation_id)
        pending = self.list_pending_approvals(consultation_id)
        if pending:
            return self._waiting_for_approval_result(consultation, pending, messages=[])

        undecided_interrupts = self._undecided_interrupt_ids(consultation_id)
        if not undecided_interrupts:
            # Nothing left to resume: the pause was fully decided and already run.
            return {
                "consultation_id": consultation_id,
                "thread_id": consultation.thread_id,
                "status": "APPROVAL_COMPLETED",
                "answer": "",
                "tool_calls": [],
                "approval_requests": [],
                "messages": [],
            }

        decisions: list[dict[str, Any]] = []
        interrupt_id = undecided_interrupts[0]
        for approval in self._interrupt_approvals(consultation_id, interrupt_id):
            decisions.append(self._decision_payload(approval))
        return self._continue_thread(consultation, decisions, interrupt_id=interrupt_id)

    def _undecided_interrupt_ids(self, consultation_id: str) -> list[str]:
        """Interrupts whose actions all carry a decision but were never executed.

        A fully rejected pause still counts as resumable: the middleware turns the
        rejection into an error ``ToolMessage`` and lets the model explain or
        replan, which is exactly the behaviour the project wants after a refusal.
        """
        ordered: list[str] = []
        by_interrupt: dict[str, list[ApprovalRequest]] = {}
        for approval in self.repository.list_approval_requests(consultation_id):
            if not approval.interrupt_id:
                continue
            by_interrupt.setdefault(approval.interrupt_id, []).append(approval)
            if approval.interrupt_id not in ordered:
                ordered.append(approval.interrupt_id)
        resumable = [
            interrupt_id
            for interrupt_id in ordered
            if all(
                approval.status is not ApprovalStatus.PENDING
                for approval in by_interrupt[interrupt_id]
            )
            and not self._interrupt_executed(consultation_id, interrupt_id)
        ]
        return resumable

    def _interrupt_approvals(
        self, consultation_id: str, interrupt_id: str
    ) -> list[ApprovalRequest]:
        approvals = [
            approval
            for approval in self.repository.list_approval_requests(consultation_id)
            if approval.interrupt_id == interrupt_id
        ]
        return sorted(approvals, key=lambda approval: approval.action_index or 0)

    def _interrupt_executed(self, consultation_id: str, interrupt_id: str) -> bool:
        """A decided interrupt counts as executed once its audit event exists."""
        return any(
            event.event_type == "APPROVAL_RESUMED"
            and event.details.get("interrupt_id") == interrupt_id
            for event in self.repository.list_audit_events(consultation_id)
        )

    @staticmethod
    def _decision_payload(approval: ApprovalRequest) -> dict[str, Any]:
        if approval.status is ApprovalStatus.REJECTED:
            return {
                "type": "reject",
                "message": approval.decision_reason
                or "顾问拒绝了该敏感操作，不要重试，除非顾问再次明确要求。",
            }
        if approval.decision is ApprovalDecision.EDIT:
            return {
                "type": "edit",
                "edited_action": {
                    "name": approval.tool_name,
                    "args": approval.edited_arguments or approval.tool_arguments,
                },
            }
        return {"type": "approve"}

    def _continue_thread(
        self,
        consultation: Consultation,
        decisions: list[dict[str, Any]],
        *,
        interrupt_id: str,
    ) -> dict[str, Any]:
        """Feed the human decisions back into the paused LangGraph thread."""
        settings = load_agent_settings()
        tools = self._build_agent_tools(ProviderSettings.from_env())

        async def run_resume():
            async with AsyncSqliteSaver.from_conn_string(
                str(self.checkpoint_path)
            ) as checkpointer:
                config = {"configurable": {"thread_id": consultation.thread_id}}
                agent = create_agent_app(
                    self.repository.database_path,
                    settings,
                    checkpointer=checkpointer,
                    tools=tools,
                )
                return await agent.ainvoke(
                    Command(resume={"decisions": decisions}), config=config
                )

        result = asyncio.run(run_resume())
        return self._finalize_agent_result(
            consultation, result, resumed_interrupt_ids=(interrupt_id,)
        )

    def _finalize_agent_result(
        self,
        consultation: Consultation,
        result: dict[str, Any],
        *,
        resumed_interrupt_ids: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Persist a finished Agent turn, or surface a fresh pause.

        ``resumed_interrupt_ids`` names the pauses this call just continued. The
        marker is written as soon as the graph leaves that pause, even when the
        very next step pauses again on a new action: otherwise the consumed pause
        would stay "resumable" and shadow every later one.
        """
        messages = result.get("messages", [])
        approvals = self._record_approval_interrupts(consultation, result)
        for interrupt_id in resumed_interrupt_ids:
            if self._interrupt_executed(consultation.id, interrupt_id):
                continue
            self.repository.add_audit_event(
                AuditEvent(
                    consultation_id=consultation.id,
                    event_type="APPROVAL_RESUMED",
                    details={
                        "thread_id": consultation.thread_id,
                        "interrupt_id": interrupt_id,
                    },
                )
            )
        if approvals:
            return self._waiting_for_approval_result(
                consultation, approvals, messages=messages
            )
        answer = _last_message_content(messages)
        tool_calls = [
            call["name"]
            for message in messages
            for call in getattr(message, "tool_calls", [])
            if call.get("name")
        ]
        self.append_message(consultation.id, "assistant", answer)
        # Capture what the search tools returned so the consultant can select a
        # version and the sensitive booking tools have something to bind to.
        version = self._record_itinerary_version(consultation, messages)
        self.repository.update_consultation_status(consultation.id, "OPEN")
        self.repository.add_audit_event(
            AuditEvent(
                consultation_id=consultation.id,
                event_type="AGENT_CHAT_COMPLETED",
                details={
                    "thread_id": consultation.thread_id,
                    "tool_calls": tool_calls,
                    "itinerary_version_id": version["version_id"] if version else None,
                },
            )
        )
        result: dict[str, Any] = {
            "consultation_id": consultation.id,
            "thread_id": consultation.thread_id,
            "status": "COMPLETED",
            "answer": answer,
            "tool_calls": tool_calls,
            "messages": [_message_as_dict(message) for message in messages],
        }
        if version is not None:
            result["itinerary_version_id"] = version["version_id"]
            result["itinerary_version"] = version["version_number"]
        return result

    def _record_itinerary_version(
        self, consultation: Consultation, messages: list[Any]
    ) -> dict[str, Any] | None:
        """Derive an itinerary version from what the Agent's search tools returned.

        Versions used to be produced by a scripted planning graph. They are now
        captured from the real conversation instead: whatever the search tools
        actually returned is what a booking may later be based on, so a version
        can never describe options the providers never offered.
        """
        observations: dict[str, dict[str, Any]] = {}
        tool_history: list[str] = []
        for message in messages:
            name = getattr(message, "name", None)
            if name not in _VERSION_TOOLS:
                continue
            payload = _loads_message_content(getattr(message, "content", None))
            if payload is None:
                continue
            tool_history.append(name)
            # A tool may run more than once in a turn (retry, refined query) and a
            # later call can come back empty or fail. Keep the first payload that
            # actually carries options rather than letting it be overwritten.
            if observations.get(name) is None and _option_list(payload):
                observations[name] = payload

        transport = _first_option(observations.get("search_transport_options"))
        hotel = _first_option(observations.get("search_hotel_options"))
        attraction_options = _option_list(observations.get("search_attractions"))
        if transport is None or hotel is None:
            return None

        requirements = consultation.trip_requirements
        traveler_count = int(requirements.get("traveler_count") or 1)
        payload = {
            "transport_option_id": transport.get("option_id"),
            "hotel_option_id": hotel.get("option_id"),
            "attraction_option_ids": [
                option.get("option_id") for option in attraction_options
            ],
            "observations": observations,
        }
        budget = _estimate_budget(
            transport, hotel, attraction_options, requirements, traveler_count
        )
        if budget is not None:
            payload["budget"] = budget
        version_id, version_number = self.repository.save_itinerary_version(
            consultation.id, payload, tool_history, observations
        )
        self.repository.add_audit_event(
            AuditEvent(
                consultation_id=consultation.id,
                event_type="ITINERARY_VERSION_RECORDED",
                details={
                    "version_id": version_id,
                    "version_number": version_number,
                    "transport_option_id": payload["transport_option_id"],
                    "hotel_option_id": payload["hotel_option_id"],
                    "tool_history": tool_history,
                },
            )
        )
        return {
            "version_id": version_id,
            "version_number": version_number,
            "payload": payload,
        }

    def append_message(self, consultation_id: str, role: str, content: str) -> None:
        self.repository.append_message(consultation_id, Message(role, content))
        self.repository.add_audit_event(
            AuditEvent(
                consultation_id=consultation_id,
                event_type="MESSAGE_APPENDED",
                details={"role": role},
            )
        )

    def update_requirements(
        self, consultation_id: str, requirements: dict[str, Any]
    ) -> None:
        self.repository.update_requirements(consultation_id, requirements)
        self.repository.add_audit_event(
            AuditEvent(
                consultation_id=consultation_id,
                event_type="REQUIREMENTS_UPDATED",
                details={"fields": sorted(requirements)},
            )
        )

    def select_itinerary(self, consultation_id: str, version_id: str) -> dict[str, Any]:
        version = self.repository.get_itinerary_version(consultation_id, version_id)
        self.repository.select_itinerary_version(consultation_id, version_id)
        self.repository.add_audit_event(
            AuditEvent(
                consultation_id=consultation_id,
                event_type="ITINERARY_SELECTED",
                details={
                    "version_id": version_id,
                    "version_number": version["version_number"],
                },
            )
        )
        return {
            "version_id": version_id,
            "version_number": version["version_number"],
            "payload": version["payload"],
        }
