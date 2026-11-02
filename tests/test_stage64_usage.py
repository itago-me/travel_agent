"""阶段 6.4：token 用量统计。

评测框架能回答"工具调了几次"，但回答不了"这一轮花了多少钱"。成本必须
先被测出来，才谈得上优化——本模块证明用量被正确采集、求和、暴露。
"""

import pytest
from langchain_core.messages import AIMessage

from travel_agent.application import create_consultation_service
from travel_agent.domain import AuditEvent
from travel_agent.service import _token_usage


class _Msg:
    """Minimal stand-in carrying only what the usage extractor reads."""

    def __init__(self, usage_metadata=None, response_metadata=None):
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata or {}


# --- 提取层 -----------------------------------------------------------------


def test_usage_sums_every_model_call_in_a_turn():
    """一轮可以有多轮工具往返，每轮一次模型调用，成本是总和。"""
    messages = [
        _Msg({"input_tokens": 1000, "output_tokens": 100}),
        _Msg({"input_tokens": 1500, "output_tokens": 200}),
        _Msg({"input_tokens": 500, "output_tokens": 50}),
    ]

    usage = _token_usage(messages)

    assert usage == {
        "input_tokens": 3000,
        "output_tokens": 350,
        "total_tokens": 3350,
        "reported": True,
    }


def test_usage_reads_openai_style_response_metadata():
    """不少 OpenAI 兼容网关只填 response_metadata.token_usage。"""
    message = _Msg(
        response_metadata={"token_usage": {"prompt_tokens": 800, "completion_tokens": 120}}
    )

    usage = _token_usage([message])

    assert usage["input_tokens"] == 800
    assert usage["output_tokens"] == 120
    assert usage["reported"] is True


def test_usage_reads_usage_key_variant():
    message = _Msg(response_metadata={"usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    assert _token_usage([message])["total_tokens"] == 15


def test_missing_usage_reports_not_reported():
    """"供应商没报" 和 "没花 token" 必须是两种不同的答案。"""
    usage = _token_usage([_Msg(), _Msg()])

    assert usage["reported"] is False
    assert usage["total_tokens"] == 0


def test_empty_message_list_is_zero_and_unreported():
    assert _token_usage([]) == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "reported": False,
    }


def test_partial_metadata_does_not_crash():
    """缺少某个 key 时按 0 处理，而不是 KeyError 打断整轮。"""
    usage = _token_usage([_Msg({"input_tokens": 42})])

    assert usage["output_tokens"] == 0
    assert usage["total_tokens"] == 42


# --- 持久化与聚合层 ---------------------------------------------------------


@pytest.fixture
def service(tmp_path):
    return create_consultation_service(tmp_path / "usage.db")


def _record_turn(service, consultation_id: str, usage: dict) -> None:
    service.repository.add_audit_event(
        AuditEvent(
            consultation_id=consultation_id,
            event_type="AGENT_CHAT_COMPLETED",
            details={"thread_id": "t", "tool_calls": [], "usage": usage},
        )
    )


def test_usage_totals_sum_across_turns(service):
    consultation = service.create_consultation("c1", "张三", "去杭州")
    _record_turn(
        service,
        consultation.id,
        {"input_tokens": 100, "output_tokens": 20, "reported": True},
    )
    _record_turn(
        service,
        consultation.id,
        {"input_tokens": 300, "output_tokens": 40, "reported": True},
    )

    totals = service.consultation_usage(consultation.id)

    assert totals == {
        "consultation_id": consultation.id,
        "turns": 2,
        "input_tokens": 400,
        "output_tokens": 60,
        "total_tokens": 460,
    }


def test_unreported_turns_do_not_inflate_turn_count(service):
    """供应商没报用量的轮次不计入 turns，否则平均数会被稀释。"""
    consultation = service.create_consultation("c1", "张三", "去杭州")
    _record_turn(service, consultation.id, {"reported": False})
    _record_turn(
        service, consultation.id, {"input_tokens": 10, "output_tokens": 2, "reported": True}
    )

    totals = service.consultation_usage(consultation.id)

    assert totals["turns"] == 1
    assert totals["total_tokens"] == 12


def test_consultation_without_turns_reports_zero(service):
    consultation = service.create_consultation("c1", "张三", "去杭州")

    totals = service.consultation_usage(consultation.id)

    assert totals["turns"] == 0
    assert totals["total_tokens"] == 0


def test_usage_isolated_between_consultations(service):
    first = service.create_consultation("c1", "张三", "去杭州")
    second = service.create_consultation("c2", "李四", "去北京")
    _record_turn(
        service, first.id, {"input_tokens": 500, "output_tokens": 50, "reported": True}
    )

    assert service.consultation_usage(first.id)["total_tokens"] == 550
    assert service.consultation_usage(second.id)["total_tokens"] == 0


def test_usage_for_unknown_consultation_raises(service):
    with pytest.raises(KeyError):
        service.consultation_usage("nope")


# --- HTTP 层 ----------------------------------------------------------------


def test_usage_endpoint_shape(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from travel_agent.api import create_app

    for provider in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{provider}_PROVIDER", "sqlite")
    client = TestClient(create_app(tmp_path / "api_usage.db"))
    created = client.post(
        "/consultations",
        json={"consultant_id": "c1", "customer_name": "张三", "message": "去杭州"},
    ).json()

    body = client.get(f"/consultations/{created['consultation_id']}/usage").json()

    assert body["turns"] == 0
    assert body["total_tokens"] == 0


def test_usage_endpoint_returns_404_for_unknown(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from travel_agent.api import create_app

    for provider in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{provider}_PROVIDER", "sqlite")
    client = TestClient(create_app(tmp_path / "api_usage2.db"))

    assert client.get("/consultations/nope/usage").status_code == 404
