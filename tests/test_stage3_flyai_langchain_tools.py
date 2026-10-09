import asyncio

from travel_agent.agent import AgentSettings, create_agent_app
from travel_agent.langchain_tools import create_travel_tools
from travel_agent.provider_factory import ProviderSettings


class RecordingFlyAIClient:
    def __init__(self):
        self.calls = []

    async def search_hotel(self, request, **filters):
        self.calls.append(("hotel", request, filters))
        return {
            "status": 0,
            "message": "success",
            "data": {
                "itemList": [
                    {
                        "shId": "hotel-1",
                        "name": "西湖测试酒店",
                        "address": "杭州市西湖区测试路1号",
                        "brandName": None,
                        "star": "舒适型",
                        "price": "¥320",
                        "rate": "4.7",
                        "mainPic": "https://example.test/hotel.jpg",
                        "detailUrl": "https://example.test/hotel/1",
                        "latitude": "30.25",
                        "longitude": "120.16",
                        "interestsPoi": "近西湖",
                    }
                ]
            },
        }

    async def search_train(self, request, **filters):
        self.calls.append(("train", request, filters))
        return {
            "status": 0,
            "message": "success",
            "data": {
                "itemList": [
                    {
                        "adultPrice": "¥553",
                        "jumpUrl": "https://example.test/train/G11",
                        "journeys": [
                            {
                                "segments": [
                                    {
                                        "depCityName": "北京",
                                        "arrCityName": "杭州",
                                        "depDateTime": "2026-10-15 08:00:00",
                                        "arrDateTime": "2026-10-15 12:28:00",
                                        "duration": "268分钟",
                                        "marketingTransportName": "中国铁路",
                                        "marketingTransportNo": "G11",
                                        "seatClassName": "二等座",
                                    }
                                ]
                            }
                        ],
                    }
                ]
            },
        }

    async def search_flight(self, request, **filters):
        raise AssertionError("train mode must not call search_flight")

    async def search_poi(self, request, **filters):
        self.calls.append(("attraction", request, filters))
        return {
            "status": 0,
            "message": "success",
            "data": {
                "itemList": [
                    {
                        "id": "poi-1",
                        "name": "西湖风景名胜区",
                        "address": "杭州市西湖区龙井路1号",
                        "freePoiStatus": "免费开放",
                        "mainPic": "https://example.test/west-lake.jpg",
                        "jumpUrl": "https://example.test/poi/1",
                        "ticketInfo": {
                            "price": None,
                            "ticketName": "西湖风景名胜区",
                        },
                    }
                ]
            },
        }


FLYAI_SETTINGS = ProviderSettings(
    hotel="flyai",
    transport="flyai",
    attraction="flyai",
)


def test_langchain_tools_invoke_flyai_providers_and_return_normalized_observations(
    tmp_path,
):
    client = RecordingFlyAIClient()
    tools = {
        tool.name: tool
        for tool in create_travel_tools(
            tmp_path / "travel.db",
            FLYAI_SETTINGS,
            flyai_client=client,
        )
    }

    transport = asyncio.run(
        tools["search_transport_options"].ainvoke(
            {
                "origin": "北京",
                "destination": "杭州",
                "departure_date": "2026-10-15",
                "traveler_count": 1,
                "mode": "train",
            }
        )
    )
    hotel = asyncio.run(
        tools["search_hotel_options"].ainvoke(
            {
                "city": "杭州",
                "check_in": "2026-10-15",
                "check_out": "2026-10-17",
                "rooms": 1,
            }
        )
    )
    attraction = asyncio.run(
        tools["search_attractions"].ainvoke(
            {"city": "杭州", "visit_date": "2026-10-16"}
        )
    )

    assert [call[0] for call in client.calls] == ["train", "hotel", "attraction"]
    assert client.calls[0][2] == {"sort_type": 3}
    assert client.calls[1][2] == {"sort": "price_asc"}
    assert transport["provider"] == "flyai_fliggy"
    assert transport["options"][0]["vehicle_type"] == "high_speed_train"
    assert hotel["provider"] == "flyai_fliggy"
    assert hotel["options"][0]["price_per_night"] == 320.0
    assert attraction["provider"] == "flyai_fliggy"
    assert attraction["options"][0]["ticket_price"] is None


def test_create_agent_receives_flyai_backed_tools(monkeypatch, tmp_path):
    client = RecordingFlyAIClient()
    captured = {}
    monkeypatch.setenv("TRAVEL_AGENT_HOTEL_PROVIDER", "flyai")
    monkeypatch.setenv("TRAVEL_AGENT_TRANSPORT_PROVIDER", "flyai")
    monkeypatch.setenv("TRAVEL_AGENT_ATTRACTION_PROVIDER", "flyai")

    class FakeChatModel:
        def __init__(self, **kwargs):
            pass

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("travel_agent.agent._import_chat_openai", lambda: FakeChatModel)
    monkeypatch.setattr("travel_agent.agent._import_create_agent", lambda: fake_create_agent)

    result = create_agent_app(
        tmp_path / "travel.db",
        AgentSettings(model_name="demo-model"),
        flyai_client=client,
    )

    tools = {tool.name: tool for tool in captured["tools"]}
    observation = asyncio.run(
        tools["search_hotel_options"].ainvoke(
            {
                "city": "杭州",
                "check_in": "2026-10-15",
                "check_out": "2026-10-17",
                "rooms": 1,
            }
        )
    )

    assert result == "compiled-agent"
    assert set(tools) == {
        "search_transport_options",
        "search_hotel_options",
        "search_attractions",
    }
    assert observation["provider"] == "flyai_fliggy"
    assert [call[0] for call in client.calls] == ["hotel"]
