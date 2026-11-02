"""阶段 6.4：HTTP 入口的契约测试。

此前 318 个测试没有一个碰 HTTP 层，结果是 api.py 里三个真实 bug（缺
``Consultation.as_dict``、错误的关键字参数 ``decision_reason``、以及把
``interrupt_id`` 传给不接受它的方法）只有在手工 curl 时才暴露。

这里用 FastAPI 的 TestClient 直接打端点，验证三件事：

1. 路由存在且状态码正确（404 必须是 404，不是 500）；
2. 响应结构包含调用方真正需要的字段；
3. 端点没有重新实现业务逻辑——它只是 service 层的一层薄封装。
"""

import pytest
from fastapi.testclient import TestClient

from travel_agent.api import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    """An API client bound to a throwaway database."""
    for provider in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{provider}_PROVIDER", "sqlite")
    app = create_app(tmp_path / "api.db")
    return TestClient(app)


# --- 建单与查单 -------------------------------------------------------------


def test_create_consultation_returns_identifiers(client):
    response = client.post(
        "/consultations",
        json={
            "consultant_id": "c1",
            "customer_name": "张三",
            "message": "我想去杭州，2人，2026年11月3日到11月5日，预算8000元",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["consultation_id"]
    assert body["thread_id"]
    assert body["status"] == "OPEN"


def test_create_consultation_extracts_requirements(client):
    """建单即抽取需求——这是产品行为，不只是写一行记录。"""
    response = client.post(
        "/consultations",
        json={
            "consultant_id": "c1",
            "customer_name": "张三",
            "message": "我想去杭州，2人，2026年11月3日到11月5日，预算8000元",
        },
    )
    consultation_id = response.json()["consultation_id"]

    body = client.get(f"/consultations/{consultation_id}").json()

    assert body["trip_requirements"]["destination"] == "杭州"
    assert body["trip_requirements"]["traveler_count"] == 2
    assert body["trip_requirements"]["budget"] == 8000.0


def test_get_consultation_reports_missing_id_as_404(client):
    """未知 id 必须是 404。500 会让调用方以为服务坏了。"""
    response = client.get("/consultations/does-not-exist")

    assert response.status_code == 404


def test_get_consultation_includes_full_state(client):
    created = client.post(
        "/consultations",
        json={"consultant_id": "c1", "customer_name": "张三", "message": "去杭州"},
    ).json()

    body = client.get(f"/consultations/{created['consultation_id']}").json()

    assert body["consultation_id"] == created["consultation_id"]
    assert body["thread_id"] == created["thread_id"]
    assert body["customer_name"] == "张三"
    assert "messages" in body
    assert "planning_constraints" in body


# --- 偏好 -------------------------------------------------------------------


def test_preferences_roundtrip(client):
    put = client.put(
        "/customers/cust-1/preferences/seat", json={"value": "靠窗"}
    )
    assert put.status_code == 200
    assert put.json() == {"ok": True}

    listed = client.get("/customers/cust-1/preferences").json()
    assert len(listed) == 1
    assert listed[0]["preference_key"] == "seat"
    assert listed[0]["value"] == "靠窗"
    assert listed[0]["source"] == "api"


def test_delete_missing_preference_is_404(client):
    response = client.delete("/customers/cust-1/preferences/nope")

    assert response.status_code == 404


def test_delete_existing_preference(client):
    client.put("/customers/cust-1/preferences/seat", json={"value": "靠窗"})

    response = client.delete("/customers/cust-1/preferences/seat")

    assert response.status_code == 200
    assert client.get("/customers/cust-1/preferences").json() == []


# --- 空集合端点 -------------------------------------------------------------


def test_empty_collections_are_empty_lists_not_404(client):
    """没有方案/审批时返回空列表，不是 404——"暂无"是正常状态。"""
    created = client.post(
        "/consultations",
        json={"consultant_id": "c1", "customer_name": "张三", "message": "去杭州"},
    ).json()
    consultation_id = created["consultation_id"]

    assert client.get(f"/consultations/{consultation_id}/itineraries").json() == []
    assert client.get(f"/consultations/{consultation_id}/approvals").json() == []


def test_itineraries_endpoint_rejects_unknown_consultation(client):
    response = client.get("/consultations/nope/itineraries")

    assert response.status_code == 404


# --- OpenAPI / 文档 ---------------------------------------------------------


def test_openapi_schema_lists_every_planned_endpoint(client):
    """schedule 6.4 要求覆盖完整闭环，缺一个都等于回到"只有两个接口"的老问题。"""
    paths = client.get("/openapi.json").json()["paths"]

    for expected in (
        "/consultations",
        "/consultations/{consultation_id}",
        "/consultations/{consultation_id}/chat",
        "/consultations/{consultation_id}/itineraries",
        "/consultations/{consultation_id}/itineraries/{version_id}/select",
        "/consultations/{consultation_id}/approvals",
        "/approvals/{approval_id}/approve",
        "/approvals/{approval_id}/reject",
        "/consultations/{consultation_id}/resume",
        "/customers/{customer_id}/preferences",
        "/customers/{customer_id}/preferences/{key}",
    ):
        assert expected in paths, f"{expected} missing from the HTTP surface"


def test_docs_are_served(client):
    assert client.get("/docs").status_code == 200


# --- 端点只是薄封装 ---------------------------------------------------------


def test_api_does_not_duplicate_business_logic(client):
    """端点应复用 service 层：需求抽取的结果与直接调 service 完全一致。

    如果 api.py 自己解析了一遍请求，这个断言会失败——那意味着两套实现
    迟早走岔（schedule 6.4 明确禁止）。
    """
    from travel_agent.requirement_extractor import extract_requirements

    message = "客户要去杭州，2人，从北京出发，2026年11月3日到11月5日，预算6000元"
    created = client.post(
        "/consultations",
        json={"consultant_id": "c1", "customer_name": "张三", "message": message},
    ).json()

    expected = extract_requirements(message)
    actual = client.get(f"/consultations/{created['consultation_id']}").json()[
        "trip_requirements"
    ]

    assert actual["destination"] == expected["destination"]
    assert actual["traveler_count"] == expected["traveler_count"]
    assert actual["budget"] == expected["budget"]


# --- 对话与 HITL 闭环（用确定性模型，不花真实 token）------------------------


class _StubModel:
    """Records tool bindings; returns plain text so no tool is ever called."""

    def __init__(self) -> None:
        self.bound_tools: list[str] = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = [getattr(t, "name", str(t)) for t in tools]
        return self

    def invoke(self, *args, **kwargs):
        raise AssertionError("chat endpoint tests must not reach a real model")

    async def ainvoke(self, *args, **kwargs):
        from langchain_core.messages import AIMessage

        return AIMessage(content="已收到您的需求，正在为您规划。")


@pytest.fixture
def client_with_model(tmp_path, monkeypatch):
    """API client whose chat endpoint is backed by a deterministic model."""
    import travel_agent.agent as agent_module

    for provider in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{provider}_PROVIDER", "sqlite")
    monkeypatch.setenv("TRAVEL_AGENT_MODEL_MODE", "openai_compatible")
    model = _StubModel()
    monkeypatch.setattr(agent_module, "_import_chat_openai", lambda: (lambda **kw: model))

    app = create_app(tmp_path / "api_chat.db")
    return TestClient(app), model


def test_chat_endpoint_completes_a_turn(client_with_model):
    client, _model = client_with_model
    created = client.post(
        "/consultations",
        json={
            "consultant_id": "c1",
            "customer_name": "张三",
            "message": "我想去杭州，2人，2026年11月3日到11月5日，预算8000元",
        },
    ).json()

    response = client.post(
        f"/consultations/{created['consultation_id']}/chat",
        json={"message": "帮我看看", "model_name": "offline"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "COMPLETED"


def test_chat_exposes_tools_to_the_model(client_with_model):
    """工具必须在 HTTP 路径上也被注册——否则模型无从查询。"""
    client, model = client_with_model
    created = client.post(
        "/consultations",
        json={"consultant_id": "c1", "customer_name": "张三", "message": "去杭州"},
    ).json()

    client.post(
        f"/consultations/{created['consultation_id']}/chat",
        json={"message": "查一下", "model_name": "offline"},
    )

    assert "search_transport_options" in model.bound_tools
    assert "search_hotel_options" in model.bound_tools


def test_chat_on_unknown_consultation_is_404(client_with_model):
    client, _model = client_with_model

    response = client.post(
        "/consultations/nope/chat", json={"message": "hi", "model_name": "offline"}
    )

    assert response.status_code == 404


def test_resume_endpoint_rejects_unknown_consultation(client):
    response = client.post("/consultations/nope/resume")

    assert response.status_code == 404


def test_approve_endpoint_rejects_unknown_approval(client):
    response = client.post(
        "/approvals/nope/approve", json={"decided_by": "c1"}
    )

    # 未知 id 必须是 404：500 会让调用方以为服务坏了
    assert response.status_code == 404


def test_select_endpoint_rejects_unknown_version(client):
    created = client.post(
        "/consultations",
        json={"consultant_id": "c1", "customer_name": "张三", "message": "去杭州"},
    ).json()

    response = client.post(
        f"/consultations/{created['consultation_id']}/itineraries/nope/select"
    )

    assert response.status_code == 404
