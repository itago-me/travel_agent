import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import date
from typing import Any, AsyncIterator, Callable, Iterator
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
from .correlation import correlated
from .persistent import PersistentLoopRunner
from .logging_utils import configure_logging, log_context, summarize
from .postgres_repository import plain_dsn
from .tracing import TurnTraceCallback
from .repository import SQLiteRepository
from .repository_protocol import RepositoryProtocol
from .tool_governance import ToolBudgetExceeded
from .requirement_extractor import extract_requirements
from .agent import AgentSettings, create_agent_app, load_agent_settings
from .langchain_tools import create_sensitive_booking_tools, create_travel_tools
from .provider_factory import ProviderSettings
from langchain_core.messages import RemoveMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command
from pydantic import ValidationError

from .booking import CancelMockBookingRequest, CreateMockBookingRequest


# The graph is only allowed to stop on these two high-risk actions. The set is
# duplicated from the middleware configuration on purpose: the service must not
# resume a pause it does not understand.
HITL_TOOL_NAMES = ("create_mock_booking", "cancel_mock_booking")


def _elapsed_ms(started: float) -> int:
    """Wall-clock milliseconds since ``started`` (from ``time.monotonic``)."""
    return int((time.monotonic() - started) * 1000)


def _last_message_content(messages: list[Any]) -> str:
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if content:
            return str(content)
        if isinstance(message, dict) and message.get("content"):
            return str(message["content"])
    return ""


def _single_event(result: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Wrap an already-computed turn result as a one-shot event stream."""
    yield {"type": "done", "result": result}


def _stream_events(chunk: Any) -> list[dict[str, Any]]:
    """Translate one LangChain message chunk into stream events.

    Only text deltas and tool-call announcements are surfaced. Reasoning
    metadata, usage chunks and empty content are dropped: they would reach the
    client as noise it cannot act on, and usage is already reported once in the
    final ``done`` payload.
    """
    events: list[dict[str, Any]] = []
    # 只转述模型自己说的话。ToolMessage 的 content 是工具返回的原始 JSON，
    # 当成 token 推给客户端会把一大坨结构化数据糊在用户脸上（实测踩到）。
    if getattr(chunk, "type", None) in ("tool", "human", "system"):
        return events
    content = getattr(chunk, "content", None)
    if isinstance(content, str) and content:
        events.append({"type": "token", "content": content})
    elif isinstance(content, list):
        # 部分模型把内容切成 [{"type":"text","text":...}] 结构
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if text:
                    events.append({"type": "token", "content": text})
    for call in getattr(chunk, "tool_call_chunks", []) or []:
        name = call.get("name") if isinstance(call, dict) else None
        if name:
            events.append({"type": "tool", "name": name})
    return events


def _token_usage(messages: list[Any]) -> dict[str, int]:
    """Sum token usage across the model messages of one turn.

    LangChain exposes ``usage_metadata`` on ``AIMessage`` for providers that
    report it. Some OpenAI-compatible gateways only populate
    ``response_metadata["token_usage"]`` (or ``usage``), so both shapes are
    read; when neither is present the counts stay zero and the caller can tell
    "provider did not report" apart from "no tokens used".

    Counting per message (not per turn) matters because a single turn can make
    several model calls - one per tool round - and the cost is the sum.
    """
    input_tokens = output_tokens = 0
    reported = False
    for message in messages:
        usage = getattr(message, "usage_metadata", None)
        if not isinstance(usage, dict):
            metadata = getattr(message, "response_metadata", None) or {}
            usage = (
                metadata.get("token_usage")
                or metadata.get("usage")
                if isinstance(metadata, dict)
                else None
            )
        if not isinstance(usage, dict):
            continue
        reported = True
        input_tokens += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        output_tokens += int(
            usage.get("output_tokens") or usage.get("completion_tokens") or 0
        )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "reported": reported,
    }


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


def _recursion_limit() -> int:
    """Upper bound on LangGraph supersteps for one turn.

    Without it a model that keeps re-querying never stops on its own. Read from
    the environment so operators can tighten it without a code change.
    """
    try:
        value = int(os.getenv("TRAVEL_AGENT_MAX_AGENT_STEPS", "25"))
    except ValueError:
        return 25
    return value if value > 0 else 25


class TurnTimeout(RuntimeError):
    """A model turn exceeded its wall-clock budget."""

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"本轮模型调用超过 {timeout_seconds:g} 秒仍未完成，已中断。"
            "请缩小问题范围后重试。"
        )


def _turn_timeout_seconds() -> float:
    """Wall-clock budget for one model turn.

    FlyAI already has timeout/retry/error grading; the model call was the only
    external dependency with no bound at all. A turn that hangs blocks the CLI
    (and, once 6.4 lands, an HTTP request) indefinitely, so it needs a ceiling.

    Default 300s: generous for a multi-tool turn on a slow provider, but finite.
    Set ``TRAVEL_AGENT_TURN_TIMEOUT_SECONDS=0`` to disable entirely.
    """
    try:
        value = float(os.getenv("TRAVEL_AGENT_TURN_TIMEOUT_SECONDS", "300"))
    except ValueError:
        return 300.0
    return value if value >= 0 else 300.0


_RUNNER: PersistentLoopRunner | None = None


class _ThreadLocks:
    """One asyncio.Lock per thread_id, created inside the resident loop.

    常驻服务下多个请求共享 Agent 与 checkpointer。同一 thread_id 的并发
    chat 会竞争 checkpoint 写入（LangGraph 假设线程内串行），因此整轮
    chat/resume 必须按 thread 串行化；不同 thread 之间不互斥，仍然并发。
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def for_thread(self, thread_id: str) -> asyncio.Lock:
        lock = self._locks.get(thread_id)
        if lock is None:
            lock = self._locks[thread_id] = asyncio.Lock()
        return lock


_THREAD_LOCKS = _ThreadLocks()


def _persistent_runner() -> PersistentLoopRunner:
    """The process-wide loop runner, created on first use."""
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = PersistentLoopRunner()
    return _RUNNER


def await_turn(coro, *, consultation_id: str | None = None):
    """Drive one async turn on the persistent loop under a wall-clock budget.

    Cancellation matters more than the error message: an abandoned model call
    keeps a socket and (with a shared loop in 6.4) a task alive. ``wait_for``
    cancels the coroutine rather than merely returning early.

    阶段 6.4：协程提交到常驻 loop 而不是每次 ``asyncio.run``。连接、信号量、
    缓存的客户端都创建和使用于同一个 loop，跨 loop 失效从根上消失。
    """
    timeout = _turn_timeout_seconds()

    async def guarded():
        if timeout <= 0:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except TimeoutError as error:
            raise TurnTimeout(timeout) from error

    # 软预算由 guarded() 内部的 wait_for 执行并转换为 TurnTimeout；
    # runner 收到的是防卡死安全网：预算 + 30s 余量。正常情况下
    # wait_for 先触发，TurnTimeout 从协程里抛出来。
    return _persistent_runner().run_coro_with_timeout(guarded(), timeout + 30)


def _agent_config(
    thread_id: str,
    consultation_id: str | None = None,
    *,
    span_recorder: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Config shared by every ainvoke/aget_state call on a thread.

    ``consultation_id`` attaches the turn tracer so model calls and tool
    executions land in the log under the same consultation, and - when
    ``span_recorder`` is supplied - in ``turn_spans`` as well. Both are optional
    because ``aget_state`` shares this config and makes no model calls.
    """
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": _recursion_limit(),
    }
    if consultation_id:
        config["callbacks"] = [
            TurnTraceCallback(consultation_id, recorder=span_recorder)
        ]
    return config


class ConsultationService:
    def __init__(
        self,
        repository: RepositoryProtocol,
        checkpoint_path: str | Path | None = None,
        *,
        agent_database: str | Path | None = None,
        checkpoint_dsn: str | None = None,
    ):
        """构造服务。

        参数分工（阶段 9 C1 的解耦结果）：

        - ``repository``：业务数据的仓储，可以是 SQLite 或 Postgres；
        - ``agent_database``：**工具数据源与 checkpointer 用的 SQLite 路径**。
          旅行查询工具读的是种子数据表（``travel_*``），仍然留在 SQLite；
          与业务仓储分开后，业务数据搬到 Postgres 时工具不必跟着改。
        - ``checkpoint_dsn``：给了它就用 Postgres checkpointer，否则用
          ``checkpoint_path`` 指向的 SQLite 文件。
        """
        # 依赖接口而非 SQLiteRepository 具体类：PostgresRepository 只要满足
        # RepositoryProtocol 就能直接接进来。
        self.repository: RepositoryProtocol = repository

        # 业务仓储可能是 Postgres（没有 database_path），所以工具用的 SQLite
        # 路径不再从仓储推导；确实没给且仓储也不是 SQLite 时明确报错，而不是
        # 在第一次调用工具时才炸。
        derived = getattr(repository, "database_path", None)
        self.agent_database = Path(agent_database or derived or "data/travel_agent.db")

        self.checkpoint_dsn = plain_dsn(checkpoint_dsn) if checkpoint_dsn else None
        self.checkpoint_path = Path(
            checkpoint_path
            or Path(self.agent_database).with_name(
                f"{Path(self.agent_database).stem}_checkpoints.sqlite"
            )
        )
        configure_logging()

    @asynccontextmanager
    async def _checkpointer(self) -> AsyncIterator[Any]:
        """LangGraph 的对话状态存储，两种后端二选一。

        Postgres saver 会自建 ``checkpoints`` / ``checkpoint_writes`` 等表——
        那是 LangGraph 的领地，**不要写进 Alembic 迁移**，它有自己的
        ``checkpoint_migrations`` 版本表。首次连接调用 ``setup()`` 建表。
        """
        if self.checkpoint_dsn:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            async with AsyncPostgresSaver.from_conn_string(
                self.checkpoint_dsn
            ) as saver:
                await saver.setup()
                yield saver
            return

        async with AsyncSqliteSaver.from_conn_string(
            str(self.checkpoint_path)
        ) as saver:
            yield saver

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
                async with self._checkpointer() as checkpointer:
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
            interrupts = _persistent_runner().run_coro(read_interrupts())
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
                self.agent_database,
                provider_settings=provider_settings,
            ),
            *create_sensitive_booking_tools(repository=self.repository),
        ]

    @correlated
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
        with log_context(
            consultation_id=consultation_id, thread_id=consultation.thread_id
        ) as log:
            log.info(
                "轮次开始 model=%s message=%s",
                model_name or loaded.model_name,
                summarize(user_message),
            )
        settings = AgentSettings(
            model_mode=loaded.model_mode,
            model_name=model_name or loaded.model_name,
            api_key=api_key or loaded.api_key,
            base_url=base_url or loaded.base_url,
        )
        provider_settings = ProviderSettings.from_env()
        tools = self._build_agent_tools(provider_settings)

        thread_lock = _THREAD_LOCKS.for_thread(consultation.thread_id)

        async def run_chat():
            # 同一 thread 的并发轮次必须串行：LangGraph checkpoint 假设
            # 线程内没有并发写入。锁在常驻 loop 内创建，按 thread 隔离，
            # 不同 thread 之间不互斥。
            async with thread_lock:
                return await _run_chat_inner()

        async def _run_chat_inner():
            async with self._checkpointer() as checkpointer:
                config = _agent_config(
                    consultation.thread_id,
                    consultation.id,
                    span_recorder=self._span_recorder(consultation.id),
                )
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
                    self.agent_database,
                    settings,
                    checkpointer=checkpointer,
                    tools=tools,
                    repository=self.repository,
                    consultation_id=consultation_id,
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

        turn_started = time.monotonic()
        try:
            result = await_turn(run_chat(), consultation_id=consultation_id)
        except ToolBudgetExceeded as exceeded:
            with log_context(consultation_id=consultation_id) as log:
                log.warning(
                    "本轮工具预算耗尽 max_calls=%s blocked_tool=%s",
                    exceeded.max_calls,
                    exceeded.tool_name,
                )
            return self._stopped_turn_result(
                consultation,
                duration_ms=_elapsed_ms(turn_started),
                status="TOOL_BUDGET_EXCEEDED",
                event_type="TURN_STOPPED_BY_BUDGET",
                answer=str(exceeded),
                details={
                    "max_calls": exceeded.max_calls,
                    "blocked_tool": exceeded.tool_name,
                },
            )
        except TurnTimeout as timed_out:
            with log_context(consultation_id=consultation_id) as log:
                log.error(
                    "本轮超时中断 timeout=%ss", timed_out.timeout_seconds
                )
            return self._stopped_turn_result(
                consultation,
                duration_ms=_elapsed_ms(turn_started),
                status="TURN_TIMEOUT",
                event_type="TURN_STOPPED_BY_TIMEOUT",
                answer=str(timed_out),
                details={"timeout_seconds": timed_out.timeout_seconds},
            )
        except GraphRecursionError as error:
            with log_context(consultation_id=consultation_id) as log:
                log.warning(
                    "本轮达步数上限 recursion_limit=%s error=%s",
                    _recursion_limit(),
                    type(error).__name__,
                )
            return self._stopped_turn_result(
                consultation,
                duration_ms=_elapsed_ms(turn_started),
                status="STEP_LIMIT_EXCEEDED",
                event_type="TURN_STOPPED_BY_STEP_LIMIT",
                answer=(
                    f"本轮已达到最大推理步数（{_recursion_limit()} 步）而未能得出结论，"
                    "已停止。请缩小问题范围后重试。"
                ),
                details={"recursion_limit": _recursion_limit(), "error": str(error)[:300]},
            )
        finalized = self._finalize_agent_result(
            consultation, result, duration_ms=_elapsed_ms(turn_started)
        )
        with log_context(
            consultation_id=consultation_id, thread_id=consultation.thread_id
        ) as log:
            log.info(
                "轮次完成 status=%s tool_calls=%s",
                finalized.get("status"),
                ",".join(finalized.get("tool_calls") or []) or "none",
            )
        return finalized

    def _stopped_turn_result(
        self,
        consultation: Consultation,
        *,
        status: str,
        event_type: str,
        answer: str,
        details: dict[str, Any],
        duration_ms: int | None = None,
    ) -> dict[str, Any]:
        """Report a governance stop as a normal result instead of an exception.

        Both a spent tool budget and a hit step limit are expected outcomes of
        running a real model. The consultant needs a usable answer and a reason,
        not a traceback.
        """
        self.repository.add_audit_event(
            AuditEvent(
                consultation_id=consultation.id,
                event_type=event_type,
                details={
                    **details,
                    # 耗时记进审计：指标端点要算延迟分位数，不能只靠日志
                    **({"duration_ms": duration_ms} if duration_ms is not None else {}),
                },
            )
        )
        self.repository.update_consultation_status(consultation.id, "OPEN")
        return {
            "consultation_id": consultation.id,
            "thread_id": consultation.thread_id,
            "status": status,
            "answer": answer,
            "tool_calls": [],
            "approval_requests": [],
            "messages": [],
        }

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

    def _span_recorder(self, consultation_id: str) -> Callable[..., None]:
        """Bind the repository span writer to one consultation."""

        def record(
            *,
            turn_id: str,
            kind: str,
            name: str,
            duration_ms: int,
            details: dict[str, Any],
            sequence: int | None = None,
        ) -> None:
            self.repository.add_turn_span(
                consultation_id=consultation_id,
                turn_id=turn_id,
                kind=kind,
                name=name,
                duration_ms=duration_ms,
                details=details,
                sequence=sequence,
            )

        return record

    def turn_spans(
        self, consultation_id: str, *, turn_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Timed steps recorded for this consultation, optionally one turn.

        Loads the consultation first so an unknown id fails loudly instead of
        looking like a turn that recorded nothing.
        """
        self.get_consultation(consultation_id)
        return self.repository.list_turn_spans(consultation_id, turn_id=turn_id)

    def metrics(self, *, window_minutes: int | None = None) -> dict[str, Any]:
        """Service-wide health numbers, aggregated from the audit trail.

        ``window_minutes`` limits the sample to recent turns - the useful
        window when asking "what is happening right now" rather than "what has
        this deployment ever done".
        """
        since = None
        if window_minutes is not None and window_minutes > 0:
            from datetime import UTC, datetime, timedelta

            since = (datetime.now(UTC) - timedelta(minutes=window_minutes)).isoformat()
        return self.repository.metrics_snapshot(since=since)

    def consultation_usage(self, consultation_id: str) -> dict[str, Any]:
        """Token usage for a consultation, aggregated from the audit trail.

        Loads the consultation first so an unknown id fails loudly instead of
        looking like a conversation that has not spent anything.
        """
        self.get_consultation(consultation_id)
        return self.repository.consult_usage_totals(consultation_id)

    def list_pending_approvals(self, consultation_id: str) -> list[ApprovalRequest]:
        """Return the pending approvals a consultant still has to decide.

        The consultation is loaded first so an unknown id fails loudly instead of
        looking like a consultation with nothing to approve.
        """
        self.get_consultation(consultation_id)
        return self.repository.list_pending_approval_requests(consultation_id)

    @correlated
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
        # 直接走内部实现：决策和它触发的执行属于同一次操作，共用 turn_id。
        return self._resume_inner(approval.consultation_id)

    @correlated
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

    @correlated
    def resume_approved_consultation(self, consultation_id: str) -> dict[str, Any]:
        """Continue an interrupted thread once every action has a decision.

        Kept as a separate entry point so a caller resuming directly gets its
        own turn id. ``decide_approval`` bypasses it via ``_resume_inner`` and
        stays in the turn it already opened - otherwise one approval split into
        two audit chains and "what did this approval cost" needed two queries.
        """
        return self._resume_inner(consultation_id)

    def _resume_inner(self, consultation_id: str) -> dict[str, Any]:
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

        resume_lock = _THREAD_LOCKS.for_thread(consultation.thread_id)

        async def run_resume():
            # 恢复同样要持 thread 锁：恢复期间不允许同线程新请求插入
            # （schedule 6.4 第 4 节的并发正确性要求）。
            async with resume_lock:
                return await _run_resume_inner()

        async def _run_resume_inner():
            async with self._checkpointer() as checkpointer:
                config = _agent_config(
                    consultation.thread_id,
                    consultation.id,
                    span_recorder=self._span_recorder(consultation.id),
                )
                agent = create_agent_app(
                    self.agent_database,
                    settings,
                    checkpointer=checkpointer,
                    tools=tools,
                    repository=self.repository,
                    consultation_id=consultation.id,
                )
                return await agent.ainvoke(
                    Command(resume={"decisions": decisions}), config=config
                )

        # 与 chat 对称：恢复也会跑完整一轮 Agent，缺了这两行则"审批后到底
        # 执行了什么"在日志里完全不可见（审计表里有，但要写 SQL 才看得到）。
        with log_context(
            consultation_id=consultation.id, thread_id=consultation.thread_id
        ) as log:
            log.info("恢复轮次开始 interrupt_id=%s", interrupt_id)
        resume_started = time.monotonic()
        result = await_turn(run_resume(), consultation_id=consultation.id)
        finalized = self._finalize_agent_result(
            consultation,
            result,
            resumed_interrupt_ids=(interrupt_id,),
            duration_ms=_elapsed_ms(resume_started),
        )
        with log_context(
            consultation_id=consultation.id, thread_id=consultation.thread_id
        ) as log:
            log.info(
                "恢复轮次完成 status=%s tool_calls=%s",
                finalized.get("status"),
                ",".join(finalized.get("tool_calls") or []) or "none",
            )
        return finalized

    def chat_consultation_stream(
        self,
        consultation_id: str,
        user_message: str,
        *,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        env_file: str | Path | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Run one agent turn, yielding events as the model produces them.

        Events are plain dicts so CLI and HTTP can format them differently:

        ``{"type": "token", "content": str}``   model text delta
        ``{"type": "tool", "name": str}``       a tool call started
        ``{"type": "done", "result": dict}``    the same payload ``chat_consultation`` returns
        ``{"type": "error", "message": str}``   the turn failed; nothing was persisted

        Streaming is deliberately limited to turns that can be computed from the
        current checkpoint in one pass. A thread with unresolved interrupts, or
        one whose saved history needs reconciliation after a crash, falls back
        to ``chat_consultation`` and emits its result as a single ``done``:
        replaying those repair paths in a second implementation would mean two
        places that must agree, which is exactly what caused earlier drift
        between the CLI and the service layer.
        """
        consultation = self.get_consultation(consultation_id)
        pending_result = self.get_pending_approval_result(consultation_id)
        if pending_result is not None:
            return _single_event(pending_result)

        loaded = load_agent_settings(env_file)
        settings = AgentSettings(
            model_mode=loaded.model_mode,
            model_name=model_name or loaded.model_name,
            api_key=api_key or loaded.api_key,
            base_url=base_url or loaded.base_url,
        )
        provider_settings = ProviderSettings.from_env()
        tools = self._build_agent_tools(provider_settings)
        thread_lock = _THREAD_LOCKS.for_thread(consultation.thread_id)
        stream_started = time.monotonic()
        with log_context(
            consultation_id=consultation_id, thread_id=consultation.thread_id
        ) as log:
            log.info(
                "流式轮次开始 model=%s message=%s",
                model_name or loaded.model_name,
                summarize(user_message),
            )

        async def _turn() -> AsyncIterator[dict[str, Any]]:
            async with thread_lock:
                async with self._checkpointer() as checkpointer:
                    config = _agent_config(
                    consultation.thread_id,
                    consultation.id,
                    span_recorder=self._span_recorder(consultation.id),
                )
                    checkpoint = await checkpointer.aget_tuple(config)
                    agent = create_agent_app(
                        self.agent_database,
                        settings,
                        checkpointer=checkpointer,
                            tools=tools,
                        repository=self.repository,
                        consultation_id=consultation.id,
                    )
                    if checkpoint is not None:
                        state = await agent.aget_state(config)
                        if getattr(state, "interrupts", ()):
                            # 未决审批：不能绕过，让调用方先决定
                            yield {
                                "type": "pause",
                                "message": "该会话有待审批的敏感操作，请先决定后再继续",
                            }
                            yield {
                                "type": "done",
                                "result": self.chat_consultation(
                                    consultation_id, user_message, model_name=model_name
                                ),
                            }
                            return
                        history_status = _tool_history_status(
                            list(state.values.get("messages", []))
                        )
                        if history_status != "complete":
                            # 崩溃后的修复路径交给经过验证的非流式实现
                            yield {
                                "type": "resync",
                                "message": "会话历史需要重建，本轮以非流式方式完成",
                            }
                            yield {
                                "type": "done",
                                "result": self.chat_consultation(
                                    consultation_id,
                                    user_message,
                                    model_name=model_name,
                                    api_key=api_key,
                                    base_url=base_url,
                                    env_file=env_file,
                                ),
                            }
                            return

                    self.append_message(consultation.id, "user", user_message)
                    current_requirements = self._absorb_requirements(
                        consultation_id, user_message
                    )
                    trip_requirements = dict(
                        self.get_consultation(consultation_id).trip_requirements
                    )
                    turn_state = {
                        "current_trip_requirements": current_requirements,
                        **self._itinerary_version_state(consultation_id),
                    }
                    if trip_requirements:
                        turn_state["trip_requirements"] = trip_requirements
                    next_input = {
                        "messages": [{"role": "user", "content": user_message}]
                    }
                    next_input.update(turn_state)
                    if checkpoint is None:
                        next_input = {
                            "messages": [
                                {"role": message.role, "content": message.content}
                                for message in self.get_consultation(
                                    consultation_id
                                ).messages
                            ],
                            **turn_state,
                        }

                    async for chunk, _metadata in agent.astream(
                        next_input, config=config, stream_mode="messages"
                    ):
                        for event in _stream_events(chunk):
                            yield event

                    final_state = await agent.aget_state(config)
                    result = dict(final_state.values)
                    interrupts = getattr(final_state, "interrupts", ())
                    if interrupts:
                        result["__interrupt__"] = interrupts
                    finalized = self._finalize_agent_result(
                        consultation,
                        result,
                        duration_ms=_elapsed_ms(stream_started),
                    )
                    with log_context(
                        consultation_id=consultation.id,
                        thread_id=consultation.thread_id,
                    ) as log:
                        log.info(
                            "流式轮次完成 status=%s tool_calls=%s",
                            finalized.get("status"),
                            ",".join(finalized.get("tool_calls") or []) or "none",
                        )
                    yield {"type": "done", "result": finalized}

        return _persistent_runner().iter_async(_turn())

    def _finalize_agent_result(
        self,
        consultation: Consultation,
        result: dict[str, Any],
        *,
        resumed_interrupt_ids: tuple[str, ...] = (),
        duration_ms: int | None = None,
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
        usage = _token_usage(messages)
        self.repository.update_consultation_status(consultation.id, "OPEN")
        self.repository.add_audit_event(
            AuditEvent(
                consultation_id=consultation.id,
                event_type="AGENT_CHAT_COMPLETED",
                details={
                    "thread_id": consultation.thread_id,
                    "tool_calls": tool_calls,
                    "itinerary_version_id": version["version_id"] if version else None,
                    # 成本记进审计：事后要能回答"这一轮花了多少 token"，
                    # 而不是只能看账单。
                    "usage": usage,
                    **({"duration_ms": duration_ms} if duration_ms is not None else {}),
                },
            )
        )
        result: dict[str, Any] = {
            "consultation_id": consultation.id,
            "thread_id": consultation.thread_id,
            "status": "COMPLETED",
            "answer": answer,
            "tool_calls": tool_calls,
            "usage": usage,
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
