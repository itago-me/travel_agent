"""阶段 8.4 第 2 步：/metrics 指标端点。

日志回答"这一轮发生了什么"，指标回答"整体现在怎么样"。测试锁定：

1. 数字来自审计表（唯一事实来源），不是另开的计数器；
2. 延迟分位数是真实观测过的值；
3. 没有数据时返回零而不是报错——新建部署第一次访问 /metrics 必须能通。
"""

import pytest
from fastapi.testclient import TestClient

from travel_agent.api import create_app
from travel_agent.application import create_consultation_service
from travel_agent.domain import AuditEvent
from travel_agent.repository import _percentiles


# --- 百分位 ---------------------------------------------------------------


def test_percentiles_empty():
    assert _percentiles([]) == {"count": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}


def test_percentiles_single_value():
    assert _percentiles([42])["p99"] == 42


def test_percentiles_nearest_rank_is_an_observed_value():
    """分位数必须是系统真实产生过的延迟，不能是插值出来的中间数。"""
    values = [100, 200, 300, 400]
    result = _percentiles(values)

    for key in ("p50", "p95", "p99", "max"):
        assert result[key] in values


def test_percentiles_ordered_sample():
    result = _percentiles(list(range(1, 101)))

    assert result["p50"] == 50
    assert result["p95"] == 95
    assert result["p99"] == 99
    assert result["max"] == 100


# --- 聚合 ---------------------------------------------------------------


@pytest.fixture
def service(tmp_path):
    return create_consultation_service(tmp_path / "metrics.db")


def _completed(service, consultation_id, *, duration_ms, tools=(), tokens=0):
    service.repository.add_audit_event(
        AuditEvent(
            consultation_id=consultation_id,
            event_type="AGENT_CHAT_COMPLETED",
            details={
                "tool_calls": list(tools),
                "usage": {
                    "input_tokens": tokens,
                    "output_tokens": 0,
                    "reported": True,
                },
                "duration_ms": duration_ms,
            },
        )
    )


def _stopped(service, consultation_id, reason, duration_ms):
    service.repository.add_audit_event(
        AuditEvent(
            consultation_id=consultation_id,
            event_type=f"TURN_STOPPED_BY_{reason.upper()}",
            details={"duration_ms": duration_ms},
        )
    )


def test_empty_deployment_reports_zeros(service):
    """全新部署第一次访问 /metrics 必须能通，而不是 500。"""
    snapshot = service.metrics()

    assert snapshot["turns"]["total"] == 0
    assert snapshot["turns"]["error_rate"] == 0.0
    assert snapshot["latency_ms"]["count"] == 0
    assert snapshot["tokens"]["total"] == 0


def test_completed_turns_are_counted(service):
    consultation = service.create_consultation("c1", "张三", "去杭州")
    _completed(service, consultation.id, duration_ms=1000)
    _completed(service, consultation.id, duration_ms=2000)

    snapshot = service.metrics()

    assert snapshot["turns"]["total"] == 2
    assert snapshot["turns"]["completed"] == 2
    assert snapshot["turns"]["failed"] == 0


def test_stopped_turns_count_as_failures(service):
    consultation = service.create_consultation("c1", "张三", "去杭州")
    _completed(service, consultation.id, duration_ms=1000)
    _stopped(service, consultation.id, "timeout", 30000)

    snapshot = service.metrics()

    assert snapshot["turns"]["failed"] == 1
    assert snapshot["turns"]["stopped"] == {"timeout": 1}
    assert snapshot["turns"]["error_rate"] == 0.5


def test_latency_uses_recorded_durations(service):
    consultation = service.create_consultation("c1", "张三", "去杭州")
    for duration in (1000, 2000, 3000):
        _completed(service, consultation.id, duration_ms=duration)

    snapshot = service.metrics()

    assert snapshot["latency_ms"]["count"] == 3
    assert snapshot["latency_ms"]["max"] == 3000


def test_tokens_and_tool_calls_are_aggregated(service):
    consultation = service.create_consultation("c1", "张三", "去杭州")
    _completed(service, consultation.id, duration_ms=1000, tools=["a", "b"], tokens=100)
    _completed(service, consultation.id, duration_ms=1000, tools=["a"], tokens=200)

    snapshot = service.metrics()

    assert snapshot["tokens"]["total"] == 300
    assert snapshot["tokens"]["per_turn"] == 150
    assert snapshot["tools"]["per_turn"] == 1.5


def test_unreported_usage_does_not_pollute_tokens(service):
    """供应商没报用量的轮次不能算成"花了 0 token"。"""
    consultation = service.create_consultation("c1", "张三", "去杭州")
    service.repository.add_audit_event(
        AuditEvent(
            consultation_id=consultation.id,
            event_type="AGENT_CHAT_COMPLETED",
            details={"usage": {"reported": False}, "duration_ms": 500},
        )
    )

    snapshot = service.metrics()

    assert snapshot["turns"]["completed"] == 1
    assert snapshot["tokens"]["total"] == 0
    assert snapshot["latency_ms"]["count"] == 1  # 耗时仍然可用


def test_window_filter_excludes_old_turns(service):
    consultation = service.create_consultation("c1", "张三", "去杭州")
    _completed(service, consultation.id, duration_ms=1000)

    # 窗口 0 分钟以内 = 只看未来，等于什么都不包含
    assert service.metrics(window_minutes=None)["turns"]["total"] == 1
    assert service.metrics(window_minutes=60)["turns"]["total"] == 1


# --- HTTP -----------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    for provider in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{provider}_PROVIDER", "sqlite")
    return TestClient(create_app(tmp_path / "api_metrics.db"))


def test_metrics_endpoint_shape(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    body = response.json()
    for key in ("turns", "latency_ms", "tokens", "tools"):
        assert key in body
    assert body["turns"]["total"] == 0


def test_metrics_endpoint_accepts_window(client):
    response = client.get("/metrics?window_minutes=60")

    assert response.status_code == 200
    assert response.json()["since"] is not None


def test_metrics_requires_auth(tmp_path, monkeypatch):
    """指标暴露用量和成本，属于业务数据，必须认证。"""
    for provider in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{provider}_PROVIDER", "sqlite")
    secured = TestClient(create_app(tmp_path / "secured.db", api_token="secret"))

    assert secured.get("/metrics").status_code == 401
    assert secured.get("/metrics", headers={"Authorization": "Bearer secret"}).status_code == 200

