import asyncio
import json
import subprocess

import pytest

from travel_agent.flyai_client import FlyAIClient
from travel_agent.flyai_errors import (
    FlyAIInvalidJsonError,
    FlyAITimeoutError,
)
from travel_agent.flyai_parsers import parse_hotel_response
from travel_agent.providers import HotelSearchRequest
from travel_agent.providers import TransportMode, TransportSearchRequest


HOTEL_RESPONSE = {
    "data": {
        "itemList": [
            {
                "address": "萧绍路950号",
                "brandName": None,
                "detailUrl": "https://example.test/hotel/67903047",
                "latitude": "30.16523",
                "longitude": "120.27861",
                "mainPic": "https://example.test/hotel.jpg",
                "name": "一期一会青年旅社(火车南站店)",
                "price": "¥23",
                "rate": None,
                "shId": "67903047",
                "star": "经济型",
            }
        ]
    },
    "message": "success",
    "status": 0,
    "systemMessage": None,
}


def test_hotel_parser_normalizes_real_flyai_response():
    result = parse_hotel_response(HOTEL_RESPONSE, city="杭州")

    assert result.provider == "flyai_fliggy"
    assert len(result.options) == 1
    option = result.options[0]
    assert option.option_id == "flyai:67903047"
    assert option.hotel_id == "67903047"
    assert option.price_per_night == 23
    assert option.currency == "CNY"
    assert option.rating is None
    assert option.available_rooms is None
    assert option.hotel_type == "经济型"
    assert option.latitude == 30.16523


def test_hotel_parser_returns_empty_result_for_empty_items():
    result = parse_hotel_response(
        {"data": {"itemList": []}, "message": "success", "status": 0},
        city="杭州",
    )

    assert result.options == []


def test_client_builds_only_supported_hotel_cli_arguments():
    client = FlyAIClient(command="flyai", timeout_seconds=12)
    request = HotelSearchRequest(
        city="杭州",
        check_in="2026-10-15",
        check_out="2026-10-17",
        rooms=1,
    )

    command = client.build_hotel_command(
        request,
        key_words="青年旅社",
        sort="price_asc",
        max_price=500,
    )

    assert command == [
        "flyai",
        "search-hotel",
        "--dest-name",
        "杭州",
        "--key-words",
        "青年旅社",
        "--sort",
        "price_asc",
        "--check-in-date",
        "2026-10-15",
        "--check-out-date",
        "2026-10-17",
        "--max-price",
        "500",
    ]
    assert "--rooms" not in command


def test_client_parses_single_line_json_with_injected_runner():
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(HOTEL_RESPONSE), "")

    client = FlyAIClient(command="flyai", runner=runner)
    request = HotelSearchRequest(
        city="杭州", check_in="2026-10-15", check_out="2026-10-17", rooms=1
    )

    raw = asyncio.run(client.search_hotel(request, sort="price_asc"))

    assert raw == HOTEL_RESPONSE
    assert calls[0][1]["timeout"] == 30


def test_client_normalizes_timeout_and_invalid_json():
    def timeout_runner(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 30)

    with pytest.raises(FlyAITimeoutError):
        asyncio.run(
            FlyAIClient(runner=timeout_runner).search_hotel(
                HotelSearchRequest(
                    city="杭州", check_in="2026-10-15", check_out="2026-10-17", rooms=1
                )
            )
        )

    def invalid_runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "not-json", "")

    with pytest.raises(FlyAIInvalidJsonError):
        asyncio.run(
            FlyAIClient(runner=invalid_runner).search_hotel(
                HotelSearchRequest(
                    city="杭州", check_in="2026-10-15", check_out="2026-10-17", rooms=1
                )
            )
        )


TRAIN_RESPONSE = {
    "data": {"itemList": [{
        "adultPrice": "¥553.0",
        "journeys": [{"segments": [{
            "depCityName": "北京", "depStationName": "北京南",
            "depDateTime": "2026-10-15 08:00:00",
            "arrCityName": "杭州", "arrStationName": "杭州东",
            "arrDateTime": "2026-10-15 12:28:00",
            "duration": "268分钟", "transportType": "火车",
            "marketingTransportNo": "G11", "seatClassName": "二等座",
        }]}],
        "jumpUrl": "https://example.test/train/G11",
    }]}, "status": 0, "message": "success"
}

FLIGHT_RESPONSE = {
    "data": {"itemList": [{
        "adultPrice": "¥400.0",
        "journeys": [{"segments": [{
            "depCityName": "北京", "depStationName": "首都国际机场",
            "depDateTime": "2026-10-15 21:00:00",
            "arrCityName": "杭州", "arrStationName": "萧山国际机场",
            "arrDateTime": "2026-10-15 23:20:00",
            "duration": "140分钟", "transportType": "飞机",
            "marketingTransportName": "国航",
            "marketingTransportNo": "CA1883", "seatClassName": "经济舱",
        }]}],
        "jumpUrl": "https://example.test/flight/CA1883",
    }]}, "status": 0, "message": "success"
}


def test_train_command_and_parser_preserve_high_speed_train_identity():
    client = FlyAIClient(command="flyai")
    request = TransportSearchRequest(origin="北京", destination="杭州", departure_date="2026-10-15", traveler_count=2, mode=TransportMode.TRAIN)
    command = client.build_train_command(request, sort_type=3, seat_class_name="二等座")
    assert command == ["flyai", "search-train", "--origin", "北京", "--destination", "杭州", "--dep-date", "2026-10-15", "--seat-class-name", "二等座", "--sort-type", "3"]

    from travel_agent.flyai_parsers import parse_transport_response
    result = parse_transport_response(TRAIN_RESPONSE, mode=TransportMode.TRAIN)
    option = result.options[0]
    assert option.mode == TransportMode.TRAIN
    assert option.vehicle_type == "high_speed_train"
    assert option.option_id == "flyai:train:G11:2026-10-15T08:00:00"
    assert option.total_price == 553
    assert option.duration_minutes == 268


def test_flight_command_and_parser_normalize_flight_fields():
    client = FlyAIClient(command="flyai")
    request = TransportSearchRequest(origin="北京", destination="杭州", departure_date="2026-10-15", traveler_count=1, mode=TransportMode.FLIGHT)
    command = client.build_flight_command(request, sort_type=3, max_price=800)
    assert command == ["flyai", "search-flight", "--origin", "北京", "--destination", "杭州", "--dep-date", "2026-10-15", "--max-price", "800", "--sort-type", "3"]

    from travel_agent.flyai_parsers import parse_transport_response
    option = parse_transport_response(FLIGHT_RESPONSE, mode=TransportMode.FLIGHT).options[0]
    assert option.mode == TransportMode.FLIGHT
    assert option.vehicle_type == "flight"
    assert option.carrier == "国航"
    assert option.total_price == 400
    assert option.duration_minutes == 140
