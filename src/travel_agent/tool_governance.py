"""工具执行治理：审计留痕与调用预算。

放在 Middleware 而不是逐个工具函数里，理由有两点：

1. **不遗漏**。新增工具自动受同一套治理约束，不需要记得给每个工具补审计；
2. **语义一致**。成功与失败用同一套事件字段，避免每个工具各写一套。

治理边界与 HITL 的分工：HITL 决定"这个动作要不要人批"，这里决定
"这次调用允许不允许发生、发生之后留下什么证据"。两者不互相替代。
"""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

# 计入预算的事件类型：成功和失败都算一次尝试，否则失败重试可以无限绕过上限。
TOOL_EVENT_TYPES = ("TOOL_EXECUTED", "TOOL_FAILED")


class ToolBudgetExceeded(RuntimeError):
    """Raised when a turn keeps calling tools after the budget was exhausted.

    The first block returns an error ``ToolMessage`` so a well-behaved model can
    wrap up gracefully. If it calls again anyway, the turn is stopped here —
    returning the message a second time would let a looping model spin until the
    graph recursion limit, which is exactly what this budget exists to prevent.
    """

    def __init__(self, consultation_id: str, max_calls: int, tool_name: str):
        self.consultation_id = consultation_id
        self.max_calls = max_calls
        self.tool_name = tool_name
        super().__init__(
            f"工具调用预算已用尽（上限 {max_calls} 次），"
            f"本次 {tool_name} 未执行，本轮已停止。"
        )


def _summarize(value: Any, *, limit: int = 300) -> str:
    """Compact argument preview for audit rows."""
    try:
        text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


class ToolGovernanceMiddleware(AgentMiddleware):
    """Record every tool execution and enforce a per-consultation call budget."""

    def __init__(
        self,
        repository: Any,
        consultation_id: str,
        *,
        max_calls: int = 30,
    ):
        self.repository = repository
        self.consultation_id = consultation_id
        self.max_calls = max_calls
        self._blocked_once = False

    # --- 审计 ---------------------------------------------------------------

    def _record(
        self,
        tool_name: str,
        event_type: str,
        details: dict[str, Any],
    ) -> None:
        from .domain import AuditEvent
        from .logging_utils import log_context

        with log_context(consultation_id=self.consultation_id) as log:
            record = log.warning if event_type == "TOOL_FAILED" else log.info
            record("工具 %s %s", tool_name, event_type)
        self.repository.add_audit_event(
            AuditEvent(
                consultation_id=self.consultation_id,
                event_type=event_type,
                details={"tool_name": tool_name, **details},
            )
        )

    def _record_success(self, tool_name: str, tool_call: dict[str, Any]) -> None:
        self._record(
            tool_name,
            "TOOL_EXECUTED",
            {"arguments": _summarize(tool_call.get("args", {}))},
        )

    def _record_failure(
        self,
        tool_name: str,
        tool_call: dict[str, Any],
        error_type: str,
        error: str,
    ) -> None:
        self._record(
            tool_name,
            "TOOL_FAILED",
            {
                "arguments": _summarize(tool_call.get("args", {})),
                "error_type": error_type,
                "error": error[:500],
            },
        )

    # --- 预算 ---------------------------------------------------------------

    # 返回审计数据rows
    def _used(self) -> int:
        return self.repository.count_tool_executions(self.consultation_id)

    def _budget_message(
        self, tool_call: dict[str, Any], tool_name: str, used: int
    ) -> ToolMessage:
        """Return an error result instead of raising.

        The model must be able to see *why* the call stopped, otherwise it will
        retry the same call and burn the remaining budget on nothing.
        """
        return ToolMessage(
            content=(
                f"工具调用预算已用尽：本轮咨询已执行 {used} 次工具调用，"
                f"上限为 {self.max_calls} 次。本次 {tool_name} 未执行。"
                "请基于已有信息给出结论，不要重复调用工具。"
            ),
            tool_call_id=tool_call.get("id") or "",
            name=tool_name,
            status="error",
        )

    def _handle_exhausted_budget(
        self, tool_call: dict[str, Any], tool_name: str, used: int
    ) -> ToolMessage:
        """Warn once, then hard-stop the turn.

        A model that ignores the warning would otherwise keep requesting tools
        until the graph recursion limit — the budget would never actually end
        the turn.
        """
        self._record(
            tool_name,
            "TOOL_BLOCKED",
            {
                "reason": "budget_exhausted",
                "used": used,
                "max_calls": self.max_calls,
                "arguments": _summarize(tool_call.get("args", {})),
            },
        )
        if self._blocked_once:
            raise ToolBudgetExceeded(self.consultation_id, self.max_calls, tool_name)
        self._blocked_once = True
        return self._budget_message(tool_call, tool_name, used)

    def _status_of(self, result: Any) -> str | None:
        return getattr(result, "status", None)

    # --- 钩子(上限) ---------------------------------------------------------------

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name") or getattr(
            getattr(request, "tool", None), "name", "unknown"
        )
        used = self._used()
        if used >= self.max_calls:
            return self._handle_exhausted_budget(tool_call, tool_name, used)
        try:
            result = handler(request)
        except Exception as error:  # noqa: BLE001 - re-raised after recording
            self._record_failure(tool_name, tool_call, type(error).__name__, str(error))
            raise
        if self._status_of(result) == "error":
            self._record_failure(
                tool_name,
                tool_call,
                "ToolError",
                str(getattr(result, "content", "")),
            )
        else:
            self._record_success(tool_name, tool_call)
        return result

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name") or getattr(
            getattr(request, "tool", None), "name", "unknown"
        )
        used = self._used()
        if used >= self.max_calls:
            return self._handle_exhausted_budget(tool_call, tool_name, used)
        try:
            result = await handler(request)
        except Exception as error:  # noqa: BLE001 - re-raised after recording
            self._record_failure(tool_name, tool_call, type(error).__name__, str(error))
            raise
        if self._status_of(result) == "error":
            self._record_failure(
                tool_name,
                tool_call,
                "ToolError",
                str(getattr(result, "content", "")),
            )
        else:
            self._record_success(tool_name, tool_call)
        return result
