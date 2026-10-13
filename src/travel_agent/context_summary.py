"""Deterministic and model-assisted conversation summaries."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


def _content(message: Any) -> str:
    value = getattr(message, "content", "")
    return str(value) if not isinstance(value, list) else str(value)


def build_structured_summary(
    messages: list[Any], requirements: dict[str, Any]
) -> dict[str, Any]:
    """Build a small, deterministic summary from known application state."""
    return {
        "requirements": dict(requirements),
        "message_count": len(messages),
        "last_message": _content(messages[-1]) if messages else "",
    }


def _summary_messages(messages: list[Any], previous_summary: str) -> list[Any]:
    transcript = "\n".join(
        f"{getattr(message, 'type', 'message')}: {_content(message)}"
        for message in messages
    )
    prompt = (
        "请将以下旅行咨询对话压缩成简短、事实准确的中文摘要。"
        "保留已确认需求、已拒绝方案、用户偏好变化和待解决问题。"
        "不要编造对话中没有出现的信息。只输出摘要正文。\n"
        f"已有摘要：{previous_summary or '无'}\n"
        f"新增对话：\n{transcript}"
    )
    return [
        SystemMessage(content="你是旅行咨询对话摘要器。"),
        HumanMessage(content=prompt),
    ]


def generate_conversation_summary_sync(
    model: Any, messages: list[Any], *, previous_summary: str = ""
) -> str | None:
    """Synchronously generate a compact summary, with safe fallback."""
    try:
        result = model.invoke(_summary_messages(messages, previous_summary))
    except Exception:
        return None
    value = _content(result).strip()
    return value or None


async def generate_conversation_summary(
    model: Any, messages: list[Any], *, previous_summary: str = ""
) -> str | None:
    """Asynchronously generate a compact summary, with safe fallback."""
    try:
        result = await model.ainvoke(_summary_messages(messages, previous_summary))
    except Exception:
        return None
    value = _content(result).strip()
    return value or None
