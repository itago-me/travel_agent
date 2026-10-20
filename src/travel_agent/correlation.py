"""Turn correlation: one id tying together everything a single turn produced.

排障的真实痛点：一轮 chat 会产生多条审计事件（TOOL_EXECUTED、
ITINERARY_VERSION_RECORDED、AGENT_CHAT_COMPLETED...），它们只靠 created_at
排序。要回答"这一轮到底发生了什么"，就得按时间戳人肉分组。

用 ``contextvars`` 而不是给每个 ``AuditEvent(...)`` 调用点传参，理由：

1. 现有 16 个写入点（含 repository 内部）不用改签名，新增的也不会漏；
2. 事件写入方（工具、中间件）不需要知道自己处于哪一轮——关联是调用
   边界（service）的职责；
3. ``contextvars`` 天然跟随 asyncio 任务传播，``asyncio.run`` 会把调用方
   的上下文带进 loop，HITL 恢复路径同样生效。
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from uuid import uuid4

_turn_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "travel_agent_turn_id", default=None
)


@contextmanager
def turn_scope():
    """Bind a fresh turn id for the duration of the block."""
    token = _turn_id.set(uuid4().hex)
    try:
        yield _turn_id.get()
    finally:
        _turn_id.reset(token)


def new_turn_id() -> str:
    """Generate a fresh turn id and bind it to the current context."""
    turn_id = uuid4().hex
    _turn_id.set(turn_id)
    return turn_id


def current_turn_id() -> str | None:
    """The turn id bound to this context, or ``None`` outside a turn."""
    return _turn_id.get()


def correlation_details() -> dict[str, str]:
    """Fields to merge into an audit event's details.

    Returns an empty dict outside a turn so startup/shutdown audits (which have
    no turn) are not decorated with a ``null`` field.
    """
    turn_id = _turn_id.get()
    return {"turn_id": turn_id} if turn_id else {}


def correlated(method):
    """Run a service method inside a fresh turn scope.

    Decorating the entry point instead of indenting its whole body keeps the
    diff to one line and guarantees the scope also covers the exception paths
    (a budget/step stop still writes its audit inside the turn).
    """
    import functools

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        with turn_scope():
            return method(*args, **kwargs)

    return wrapper
