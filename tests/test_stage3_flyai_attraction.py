import asyncio
import json
import subprocess

from travel_agent.flyai_client import FlyAIClient
from travel_agent.flyai_parsers import parse_attraction_response
from travel_agent.providers import AttractionSearchRequest


POI_RESPONSE = {
    "data": {
        "itemList": [
            {
                "address": "杭州市西湖区龙井路1号",
                "id": "177224",
                "mainPic": "https://example.test/west-lake.jpg",
                "jumpUrl": "https://example.test/poi/177224",
                "name": "西湖风景名胜区",
                "freePoiStatus": "免费开放",
                "ticketInfo": {
                    "price": None,
                    "priceDate": "2026-10-15",
                    "ticketName": "西湖风景名胜区",
                },
            },
            {
                "address": "杭州市西湖区法云弄1号",
                "id": "177225",
                "mainPic": None,
                "jumpUrl": "https://example.test/poi/177225",
                "name": "灵隐寺",
                "ticketInfo": {
                    "price": "¥75",
                    "priceDate": "2026-10-15",
                    "ticketName": "灵隐寺成人票",
                },
            },
        ]
    },
    "message": "success",
    "status": 0,
    "systemMessage": None,
}


def test_poi_command_uses_only_supported_flyai_arguments():
    client = FlyAIClient(command="flyai")
    request = AttractionSearchRequest(city="杭州", visit_date="2026-10-15")

    command = client.build_poi_command(
        request,
        poi_level=5,
        keyword="西湖",
        category="山湖田园",
    )

    assert command == [
        "flyai",
        "search-poi",
        "--city-name",
        "杭州",
        "--poi-level",
        "5",
        "--keyword",
        "西湖",
        "--category",
        "山湖田园",
    ]
    assert "--visit-date" not in command


def test_poi_search_executes_cli_and_returns_json():
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(POI_RESPONSE), "")

    raw = asyncio.run(
        FlyAIClient(runner=runner).search_poi(
            AttractionSearchRequest(city="杭州", visit_date="2026-10-15"),
            keyword="西湖",
        )
    )

    assert raw == POI_RESPONSE
    assert calls[0][1] == "search-poi"


def test_poi_parser_preserves_unknown_duration_and_optional_ticket_price():
    result = parse_attraction_response(
        POI_RESPONSE,
        city="杭州",
        visit_date="2026-10-15",
    )

    assert result.provider == "flyai_fliggy"
    assert len(result.options) == 2
    free, paid = result.options
    assert free.option_id == "flyai:poi:177224"
    assert free.ticket_price is None
    assert free.duration_minutes is None
    assert free.address == "杭州市西湖区龙井路1号"
    assert free.free_status == "免费开放"
    assert paid.ticket_price == 75
    assert paid.currency == "CNY"
    assert paid.ticket_name == "灵隐寺成人票"


def test_poi_parser_returns_empty_result_for_empty_items():
    result = parse_attraction_response(
        {"data": {"itemList": []}, "status": 0, "message": "success"},
        city="杭州",
        visit_date="2026-10-15",
    )

    assert result.options == []
