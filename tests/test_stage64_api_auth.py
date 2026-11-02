"""阶段 6.4：HTTP 入口的 bearer 认证。

未认证的端点能花掉真实的模型预算（`/chat` 会调用上游 LLM），因此
"认证是否真的挡住了请求"必须由测试证明，而不是由代码阅读证明。
"""

import pytest
from fastapi.testclient import TestClient

from travel_agent.api import create_app

TOKEN = "test-secret-token"


@pytest.fixture
def secured(tmp_path, monkeypatch):
    for provider in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{provider}_PROVIDER", "sqlite")
    return TestClient(create_app(tmp_path / "auth.db", api_token=TOKEN))


@pytest.fixture
def open_api(tmp_path, monkeypatch):
    """Token unset -> auth disabled (localhost development mode)."""
    monkeypatch.delenv("TRAVEL_AGENT_API_TOKEN", raising=False)
    for provider in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{provider}_PROVIDER", "sqlite")
    return TestClient(create_app(tmp_path / "open.db"))


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def test_business_endpoint_rejects_missing_token(secured):
    response = secured.post(
        "/consultations",
        json={"consultant_id": "c1", "customer_name": "张三", "message": "去杭州"},
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_business_endpoint_rejects_wrong_token(secured):
    response = secured.get("/customers/c1/preferences", headers={"Authorization": "Bearer nope"})

    assert response.status_code == 401


def test_business_endpoint_rejects_non_bearer_scheme(secured):
    """Basic 认证或裸 token 都不该被接受——只认 Bearer。"""
    response = secured.get("/customers/c1/preferences", headers={"Authorization": TOKEN})

    assert response.status_code == 401


def test_correct_token_grants_access(secured):
    response = secured.post(
        "/consultations",
        json={"consultant_id": "c1", "customer_name": "张三", "message": "去杭州"},
        headers=_auth(),
    )

    assert response.status_code == 200
    assert response.json()["consultation_id"]


def test_every_business_route_is_guarded(secured):
    """逐条验证：新增端点忘了挂依赖，这个测试会失败。"""
    schema = secured.get("/openapi.json").json()["paths"]
    open_paths = {"/health", "/openapi.json", "/docs", "/redoc"}

    for path, methods in schema.items():
        if path in open_paths:
            continue
        for method in methods:
            response = secured.request(method.upper(), path.replace("{", "").replace("}", ""))
            assert response.status_code in (401, 422), (
                f"{method.upper()} {path} 未受保护，返回 {response.status_code}"
            )


def test_health_is_open(secured):
    """探针必须免认证，否则负载均衡无法检查存活。"""
    response = secured.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "auth": "bearer"}


def test_docs_are_open(secured):
    assert secured.get("/docs").status_code == 200


def test_chat_requires_token_before_touching_the_model(secured):
    """最贵的端点：认证必须在调用模型之前失败。"""
    response = secured.post(
        "/consultations/any-id/chat", json={"message": "hi", "model_name": "offline"}
    )

    assert response.status_code == 401


def test_no_token_configured_leaves_api_open(open_api):
    """未配置 token 时保持开发模式可用——但这是 localhost 专用形态。"""
    response = open_api.post(
        "/consultations",
        json={"consultant_id": "c1", "customer_name": "张三", "message": "去杭州"},
    )

    assert response.status_code == 200
    assert open_api.get("/health").json()["auth"] == "none"


def test_token_from_environment_is_honoured(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAVEL_AGENT_API_TOKEN", "from-env")
    for provider in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{provider}_PROVIDER", "sqlite")
    client = TestClient(create_app(tmp_path / "env.db"))

    assert client.get("/customers/c1/preferences").status_code == 401
    assert (
        client.get(
            "/customers/c1/preferences", headers={"Authorization": "Bearer from-env"}
        ).status_code
        == 200
    )
