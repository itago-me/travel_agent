"""阶段 6.2 D：模型调用超时与结构化日志。

现状问题：FlyAI 侧已有完整韧性（超时、重试、错误分级），但**模型调用是唯一
没有超时控制的外部依赖**。一次卡死的模型会让 CLI 永久挂起；在 6.4 的 HTTP
形态下更是直接占死一个请求。

日志方面，`service.py` / `agent.py` 此前没有任何 logging。

关于顺序：原规划把本项排在 6.4 常驻服务之后，理由是"超时需要可取消的 loop"。
实测 `asyncio.wait_for` 在 `asyncio.run` 内部能正常超时并取消，**没有技术
阻塞**，因此提前实现，在 6.4 之前就消除风险。
"""

import time

import pytest

from travel_agent.application import create_consultation_service
from travel_agent.correlation import turn_scope
from travel_agent.logging_utils import (
    configure_logging,
    log_context,
    logger,
    summarize,
)
from travel_agent.service import TurnTimeout, _turn_timeout_seconds, await_turn


def make(tmp_path, name="t"):
    service = create_consultation_service(tmp_path / f"{name}.db")
    consultation = service.create_consultation("c1", "客户", "去杭州")
    return service, consultation


# --- 超时配置 ---------------------------------------------------------------


def test_timeout_defaults_to_300_seconds(monkeypatch):
    monkeypatch.delenv("TRAVEL_AGENT_TURN_TIMEOUT_SECONDS", raising=False)
    assert _turn_timeout_seconds() == 300.0


def test_timeout_is_configurable(monkeypatch):
    monkeypatch.setenv("TRAVEL_AGENT_TURN_TIMEOUT_SECONDS", "12.5")
    assert _turn_timeout_seconds() == 12.5


def test_invalid_timeout_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("TRAVEL_AGENT_TURN_TIMEOUT_SECONDS", "not-a-number")
    assert _turn_timeout_seconds() == 300.0


def test_zero_timeout_disables_the_budget(monkeypatch):
    monkeypatch.setenv("TRAVEL_AGENT_TURN_TIMEOUT_SECONDS", "0")
    assert _turn_timeout_seconds() == 0


# --- 超时真的会中断 ---------------------------------------------------------


def test_await_turn_raises_on_a_hanging_coroutine(monkeypatch):
    import asyncio

    monkeypatch.setenv("TRAVEL_AGENT_TURN_TIMEOUT_SECONDS", "0.2")

    async def hang():
        await asyncio.sleep(30)

    started = time.monotonic()
    with pytest.raises(TurnTimeout):
        await_turn(hang())
    elapsed = time.monotonic() - started

    assert elapsed < 5, "必须真正中断，而不是等协程自己结束"


def test_await_turn_passes_through_fast_results(monkeypatch):
    monkeypatch.setenv("TRAVEL_AGENT_TURN_TIMEOUT_SECONDS", "5")

    async def quick():
        return {"ok": True}

    assert await_turn(quick()) == {"ok": True}


def test_timeout_message_names_the_budget(monkeypatch):
    error = TurnTimeout(45)
    assert "45" in str(error)
    assert "中断" in str(error)


def test_a_hanging_model_becomes_a_normal_result(monkeypatch, tmp_path):
    """端到端：模型卡死时顾问得到可读结论，而不是挂起或 traceback。"""
    from langchain_core.language_models.chat_models import BaseChatModel

    from travel_agent.agent import AgentSettings

    class HangingModel(BaseChatModel):
        @property
        def _llm_type(self):
            return "hang"

        def bind_tools(self, tools, **kwargs):
            return self

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            import asyncio

            await asyncio.sleep(30)

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise AssertionError("不应走到同步路径")

    service, consultation = make(tmp_path)
    monkeypatch.setattr("travel_agent.agent.create_chat_model", lambda _s: HangingModel())
    monkeypatch.setattr(
        "travel_agent.service.load_agent_settings",
        lambda *_: AgentSettings(
            model_mode="openai_compatible", model_name="demo", api_key=None, base_url=None
        ),
    )
    monkeypatch.setenv("TRAVEL_AGENT_TURN_TIMEOUT_SECONDS", "0.3")

    result = service.chat_consultation(consultation.id, "查询")

    assert result["status"] == "TURN_TIMEOUT"
    assert "秒" in result["answer"]
    events = [
        event.event_type
        for event in service.repository.list_audit_events(consultation.id)
    ]
    assert "TURN_STOPPED_BY_TIMEOUT" in events
    # 超时后咨询必须仍可继续操作
    assert service.get_consultation(consultation.id).status == "OPEN"


# --- 日志 -------------------------------------------------------------------


def test_configure_logging_is_idempotent():
    configure_logging("info")
    handlers_after_first = len(logger.handlers)
    configure_logging("info")

    assert len(logger.handlers) == handlers_after_first, "重复调用不应叠加 handler"


def test_log_context_renders_fields():
    configure_logging("info")
    with log_context(consultation_id="c-1", thread_id="th-1") as log:
        rendered, _ = log.process("轮次开始", {})

    assert "consultation_id=c-1" in rendered
    assert "thread_id=th-1" in rendered


def test_log_context_picks_up_the_current_turn():
    """日志与审计必须对同一轮次使用同一个 id。"""
    configure_logging("info")
    with turn_scope() as turn_id:
        with log_context(consultation_id="c-1") as log:
            rendered, _ = log.process("msg", {})

    assert f"turn_id={turn_id}" in rendered


def test_log_context_omits_missing_fields():
    with log_context(consultation_id="c-1", thread_id=None) as log:
        rendered, _ = log.process("msg", {})

    assert "thread_id" not in rendered


def test_summarize_truncates_long_content():
    assert summarize("短内容") == "短内容"
    long_text = "x" * 500
    rendered = summarize(long_text)
    assert len(rendered) < len(long_text)
    assert "500 chars" in rendered


def test_summarize_handles_none():
    assert summarize(None) == "<none>"
