"""Structured logging for the consultation service.

此前 ``service.py`` / ``agent.py`` 完全没有日志——真实模型跑一轮可能耗时
数十秒、调用十几次工具，失败时只能看到最终异常，看不到"卡在哪一步"。

设计约束：

1. **只用标准库 logging**，不引入依赖；默认挂在 ``travel_agent`` logger 下，
   由使用方通过 ``logging.basicConfig()`` 或 ``TRAVEL_AGENT_LOG_LEVEL`` 控制。
2. **字段固定**：``consultation_id`` / ``thread_id`` / ``turn_id`` 是排障必需的
   三个维度，统一由 ``log_context()`` 注入，避免每个调用点手写。
3. **不带业务敏感内容**：不记录客户姓名、完整消息体，只记长度或摘要。
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any

from .correlation import current_turn_id

LOGGER_NAME = "travel_agent"
logger = logging.getLogger(LOGGER_NAME)

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def configure_logging(level: str | None = None) -> None:
    """Attach a stderr handler once, honouring ``TRAVEL_AGENT_LOG_LEVEL``.

    Idempotent: repeated calls (every service construction) do not stack
    handlers, which would otherwise duplicate every line.
    """
    if logger.handlers:
        return
    raw = (level or os.getenv("TRAVEL_AGENT_LOG_LEVEL") or "warning").strip().lower()
    logger.setLevel(_LEVELS.get(raw, logging.WARNING))
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    # 不让日志冒泡到 root，避免与使用方自己的配置重复输出
    logger.propagate = False


class _ContextAdapter(logging.LoggerAdapter):
    """Render context fields as ``key=value`` pairs after the message."""

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        fields = {k: v for k, v in self.extra.items() if v} if self.extra else {}
        if not fields:
            return msg, kwargs
        rendered = " ".join(f"{key}={value}" for key, value in fields.items())
        return f"{msg} | {rendered}", kwargs


@contextmanager
def log_context(**fields: Any):
    """Bind consultation/thread/turn fields for the duration of the block.

    ``turn_id`` is filled from the correlation context when not passed, so a
    log line and the audit rows it describes always agree on the turn.
    """
    merged = {k: v for k, v in fields.items() if v is not None}
    merged.setdefault("turn_id", current_turn_id())
    yield _ContextAdapter(logger, merged)


def summarize(text: Any, limit: int = 80) -> str:
    """Shorten content for logs: length plus a prefix, never the whole body."""
    if text is None:
        return "<none>"
    value = str(text)
    if len(value) <= limit:
        return value
    return f"{value[:limit]}…({len(value)} chars)"
