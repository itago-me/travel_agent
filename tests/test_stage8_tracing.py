"""阶段 8.4 第 1 步：turn 级日志埋点。

把「轮次开始 → 轮次完成」之间那段空白变成时间线。测试锁定三件事：

1. 每次模型调用、每次工具执行都产生一行带耗时的日志；
2. 日志带 turn_id，能和审计事件对应上；
3. 失败与「未报告」有自己的措辞，不会被当成 0 或假装成功。
"""

import logging

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult

from travel_agent.tracing import TurnTraceCallback, _extract_usage, _result_count


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


@pytest.fixture
def captured():
    """Capture travel_agent log lines for the duration of a test."""
    from travel_agent.logging_utils import configure_logging, logger

    configure_logging("info")
    handler = _Capture()
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


# --- token 提取 -------------------------------------------------------------


def test_usage_from_generation_metadata():
    """现代 provider 把 usage 挂在 message 上，llm_output 常是 None。"""
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="x",
                        usage_metadata={
                            "input_tokens": 800,
                            "output_tokens": 50,
                            "total_tokens": 850,
                        },
                    )
                )
            ]
        ]
    )

    assert _extract_usage(response) == {"input": 800, "output": 50}


def test_usage_from_llm_output():
    response = LLMResult(
        generations=[],
        llm_output={"token_usage": {"prompt_tokens": 100, "completion_tokens": 20}},
    )

    assert _extract_usage(response) == {"input": 100, "output": 20}


def test_missing_usage_reports_minus_one():
    """"未报告" 必须是 -1，不能是 0——0 会被读成"这次调用免费"。"""
    response = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x"))]])

    assert _extract_usage(response) == {"input": -1, "output": -1}


# --- 结果条数 ---------------------------------------------------------------


def test_result_count_from_tool_message():
    """on_tool_end 实际收到 ToolMessage（实测），不是原始返回值。"""
    message = ToolMessage(content='{"options": [{}, {}, {}]}', tool_call_id="c1")

    assert _result_count(message) == 3


def test_result_count_from_plain_dict():
    assert _result_count({"options": [{}, {}]}) == 2


def test_result_count_absent_when_tool_has_no_options():
    """非搜索类工具没有 options 数组，不该编造一个数字。"""
    assert _result_count({"booking_id": "b1"}) is None
    assert _result_count(None) is None
    assert _result_count("not json") is None


# --- 回调日志 ---------------------------------------------------------------


def test_model_call_logs_duration_and_tokens(captured):
    callback = TurnTraceCallback("consult-1")
    callback.on_chat_model_start({}, [], run_id="r1")
    callback.on_llm_end(
        LLMResult(
            generations=[
                [
                    ChatGeneration(
                        message=AIMessage(
                            content="x",
                            usage_metadata={
                                "input_tokens": 900,
                                "output_tokens": 60,
                                "total_tokens": 960,
                            },
                        )
                    )
                ]
            ]
        ),
        run_id="r1",
    )

    line = next(l for l in captured.lines if "模型调用" in l)
    assert "#1" in line
    assert "input=900" in line
    assert "output=60" in line
    assert "duration_ms=" in line


def test_tool_call_logs_duration_and_result_count(captured):
    callback = TurnTraceCallback("consult-1")
    callback.on_tool_start({"name": "search_hotel_options"}, "{}", run_id="t1")
    callback.on_tool_end(
        ToolMessage(content='{"options": [{}, {}]}', tool_call_id="c1"), run_id="t1"
    )

    line = next(l for l in captured.lines if "search_hotel_options" in l)
    assert "duration_ms=" in line
    assert "result_count=2" in line


def test_tool_error_logs_warning_with_error_type(captured):
    callback = TurnTraceCallback("consult-1")
    callback.on_tool_start({"name": "search_hotel_options"}, "{}", run_id="t1")
    callback.on_tool_error(TimeoutError("boom"), run_id="t1")

    line = next(l for l in captured.lines if "search_hotel_options" in l)
    assert "失败" in line
    assert "TimeoutError" in line


def test_model_error_logs_error_level(captured):
    callback = TurnTraceCallback("consult-1")
    callback.on_chat_model_start({}, [], run_id="r1")
    callback.on_llm_error(RuntimeError("upstream"), run_id="r1")

    line = next(l for l in captured.lines if "模型调用" in l)
    assert "失败" in line
    assert "RuntimeError" in line


def test_model_calls_are_numbered_in_order(captured):
    """一轮里有多次模型调用，编号让"第几次"可读。"""
    callback = TurnTraceCallback("consult-1")
    response = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="x", usage_metadata={
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                        }))]]
    )
    for index in range(3):
        callback.on_chat_model_start({}, [], run_id=f"r{index}")
        callback.on_llm_end(response, run_id=f"r{index}")

    numbered = [l for l in captured.lines if "模型调用" in l]
    assert len(numbered) == 3
    assert "#1" in numbered[0] and "#2" in numbered[1] and "#3" in numbered[2]


def test_callback_does_not_log_tool_input(captured):
    """工具输入可能含敏感信息，日志只记名字和耗时。"""
    callback = TurnTraceCallback("consult-1")
    callback.on_tool_start(
        {"name": "search_hotel_options"}, '{"city": "杭州", "secret": "x"}', run_id="t1"
    )
    callback.on_tool_end({"options": []}, run_id="t1")

    joined = "\n".join(captured.lines)
    assert "secret" not in joined
    assert "杭州" not in joined
