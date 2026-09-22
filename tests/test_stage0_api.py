from fastapi.testclient import TestClient

from travel_agent.api import create_app


def test_create_consultation_api_returns_consultation_identity(tmp_path):
    client = TestClient(create_app(tmp_path / "travel_agent.db"))

    response = client.post(
        "/consultations",
        json={
            "consultant_id": "consultant-1",
            "customer_name": "赵六",
            "initial_message": "想去西安五天",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["consultation_id"]
    assert body["thread_id"]
    assert body["status"] == "OPEN"


def test_get_consultation_api_returns_persisted_messages(tmp_path):
    client = TestClient(create_app(tmp_path / "travel_agent.db"))
    created = client.post(
        "/consultations",
        json={
            "consultant_id": "consultant-1",
            "customer_name": "钱七",
            "initial_message": "想去青岛",
        },
    ).json()

    response = client.get(f"/consultations/{created['consultation_id']}")

    assert response.status_code == 200
    assert response.json()["messages"] == [{"role": "user", "content": "想去青岛"}]


def test_get_unknown_consultation_returns_not_found(tmp_path):
    client = TestClient(create_app(tmp_path / "travel_agent.db"))

    response = client.get("/consultations/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"detail": "Consultation not found: does-not-exist"}
