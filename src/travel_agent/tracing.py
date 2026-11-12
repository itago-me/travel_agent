"""Turn-level tracing: what happened between "轮次开始" and "轮次完成".

阶段 8.4 第 1 步。现状是每轮只有两行日志，中间 4~15 秒是黑盒——模型调了几次、
每次多少 token、哪个工具慢、有没有重试，全都看不到。这一层把那段空白变成
一条时间线：

    轮次开始
    模型调用 #1 duration_ms=1200 input=800 output=50
    工具 search_hotel duration_ms=120 result_count=2
    模型调用 #2 duration_ms=3400 input=1500 output=300
    轮次完成

**用 LangChain 回调而不是改 Agent 代码**：回调是框架提供的观测点，
on_chat_model_start / on_llm_end / on_tool_start / on_tool_end 恰好覆盖
需要的四个时刻，不需要在业务逻辑里插桩，也不会因为模型或工具实现变化而失效。

**为什么不用 OpenTelemetry**：单个进程、单个服务，手写 span 已经够用，而
OTel 要额外理解 span 上下文传播与 exporter。这里的事件形状与将来迁移到
OTel 时一致（一个环节 = 一个带耗时的 span），所以现在做不会白做。

**不记敏感内容**：只记工具名、耗时、token 数和错误类型，不记工具输入输出
正文——与 logging_utils 的既有约束一致。
"""

from __future__ import annotations

import time
from typing import Any, Callable

from langchain_core.callbacks import BaseCallbackHandler

from .correlation import current_turn_id
from .logging_utils import log_context, logger


def _extract_usage(response: Any) -> dict[str, int]:
    """Pull token counts out of an ``LLMResult``.

    Providers differ: some fill ``llm_output``, most modern ones attach
    ``usage_metadata`` to the message inside each generation. When neither is
    present the counts stay -1 so a log line can say "not reported" instead of
    pretending the call was free.
    """
    usage = {"input": -1, "output": -1}
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        raw = llm_output.get("token_usage") or llm_output.get("usage")
        if isinstance(raw, dict):
            input_tokens = raw.get("input_tokens") or raw.get("prompt_tokens")
            output_tokens = raw.get("output_tokens") or raw.get("completion_tokens")
            if input_tokens is not None:
                usage["input"] = int(input_tokens)
            if output_tokens is not None:
                usage["output"] = int(output_tokens)
            if usage["input"] >= 0 or usage["output"] >= 0:
                return usage
    for generation_list in getattr(response, "generations", ()) or ():
        for generation in generation_list:
            metadata = getattr(getattr(generation, "message", None), "usage_metadata", None)
            if isinstance(metadata, dict):
                usage["input"] = int(metadata.get("input_tokens") or 0)
                usage["output"] = int(metadata.get("output_tokens") or 0)
                return usage
    return usage


def _result_count(output: Any) -> int | None:
    """How many options a search tool returned, when it says so.

    Tool output is a JSON string with an ``options`` array for every travel
    search tool. Counting it turns "the tool ran" into "the tool ran and found
    2 hotels" - which is what distinguishes an empty upstream from a broken
    call.
    """
    import json

    # on_tool_end 实际收到的是 ToolMessage（实测），它在 .content 里装着
    # 工具返回值；直接调用工具时则是 dict。四种形态都见过：
    # ToolMessage / JSON 字符串 / dict / None。
    payload: Any = getattr(output, "content", output)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return None
    if not isinstance(payload, dict):
        return None
    options = payload.get("options")
    return len(options) if isinstance(options, list) else None


class TurnTraceCallback(BaseCallbackHandler):
    """Log one line per model call and per tool execution within a turn.

    Attached through the invoke config so it sees exactly the calls the agent
    makes - no instrumentation inside the agent itself.
    """

    def __init__(
        self,
        consultation_id: str,
        *,
        recorder: Callable[..., None] | None = None,
    ) -> None:
        """``recorder`` receives ``(turn_id, kind, name, duration_ms, details)``.

        Logging alone loses the breakdown when the process exits; passing a
        recorder persists the same numbers so "why was that turn slow" is still
        answerable tomorrow.
        """
        self.consultation_id = consultation_id
        self.recorder = recorder
        self._model_calls = 0
        self._span_sequence = 0
        self._model_starts: dict[Any, float] = {}
        self._tool_starts: dict[Any, tuple[str, float]] = {}

    def _record(
        self,
        kind: str,
        name: str,
        duration_ms: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Persist one span; a recording failure must never break the turn."""
        if self.recorder is None or duration_ms < 0:
            return
        turn_id = current_turn_id()
        if not turn_id:
            return
        # 回调都在同一个 loop 线程上触发，普通自增即可；并发工具调用的
        # span 各自拿到不同序号，不需要加锁。
        self._span_sequence += 1
        try:
            self.recorder(
                turn_id=turn_id,
                kind=kind,
                name=name,
                duration_ms=duration_ms,
                details=details or {},
                sequence=self._span_sequence,
            )
        except Exception:  # noqa: BLE001 - 观测失败不该影响业务
            logger.debug("span 记录失败", exc_info=True)

    # --- 模型调用 ---------------------------------------------------------

    def on_chat_model_start(self, serialized, messages, **kwargs: Any) -> None:
        self._begin_model_call(kwargs.get("run_id"))

    def on_llm_start(self, serialized, prompts, **kwargs: Any) -> None:
        # 非 chat 模型走这个钩子；chat 模型只会触发 on_chat_model_start。
        self._begin_model_call(kwargs.get("run_id"))

    def _begin_model_call(self, run_id: Any) -> None:
        self._model_calls += 1
        self._model_starts[run_id] = time.monotonic()

    def on_llm_end(self, response, **kwargs: Any) -> None:
        started = self._model_starts.pop(kwargs.get("run_id"), None)
        duration_ms = int((time.monotonic() - started) * 1000) if started else -1
        usage = _extract_usage(response)
        with log_context(consultation_id=self.consultation_id) as log:
            log.info(
                "模型调用 #%d duration_ms=%d input=%s output=%s",
                self._model_calls,
                duration_ms,
                usage["input"] if usage["input"] >= 0 else "未报告",
                usage["output"] if usage["output"] >= 0 else "未报告",
            )
        self._record(
            "model",
            f"模型调用 #{self._model_calls}",
            duration_ms,
            {
                "index": self._model_calls,
                "input_tokens": usage["input"],
                "output_tokens": usage["output"],
            },
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        started = self._model_starts.pop(kwargs.get("run_id"), None)
        duration_ms = int((time.monotonic() - started) * 1000) if started else -1
        with log_context(consultation_id=self.consultation_id) as log:
            log.error(
                "模型调用 #%d 失败 duration_ms=%d error=%s",
                self._model_calls,
                duration_ms,
                type(error).__name__,
            )

    # --- 工具执行 ---------------------------------------------------------

    def on_tool_start(self, serialized, input_str, **kwargs: Any) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "<unknown>"
        self._tool_starts[kwargs.get("run_id")] = (str(name), time.monotonic())

    def on_tool_end(self, output, **kwargs: Any) -> None:
        name, started = self._tool_starts.pop(
            kwargs.get("run_id"), (kwargs.get("name") or "<unknown>", None)
        )
        duration_ms = int((time.monotonic() - started) * 1000) if started else -1
        count = _result_count(output)
        with log_context(consultation_id=self.consultation_id) as log:
            if count is None:
                log.info("工具 %s duration_ms=%d", name, duration_ms)
            else:
                log.info("工具 %s duration_ms=%d result_count=%d", name, duration_ms, count)
        self._record(
            "tool",
            name,
            duration_ms,
            {} if count is None else {"result_count": count},
        )

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        name, started = self._tool_starts.pop(
            kwargs.get("run_id"), (kwargs.get("name") or "<unknown>", None)
        )
        duration_ms = int((time.monotonic() - started) * 1000) if started else -1
        with log_context(consultation_id=self.consultation_id) as log:
            log.warning(
                "工具 %s 失败 duration_ms=%d error=%s",
                name,
                duration_ms,
                type(error).__name__,
            )
        self._record("tool", name, duration_ms, {"error": type(error).__name__})
