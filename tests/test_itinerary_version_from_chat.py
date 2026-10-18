"""方案版本改为由真实对话产生，而不是脚本化的 planning 图。

旧实现用 MockPlanningModel + 规划图生成 itinerary version，与真实 Agent 无关：
图里跑的工具调用和顾问实际看到的对话可以完全不一致。现在版本直接从 chat 的
ToolMessage 里提取——搜索工具真正返回了什么，版本就记录什么，因此版本不可能
描述供应商从未提供过的选项。
"""

import json
import pathlib

import pytest
from langchain_core.messages import ToolMessage

from travel_agent.application import create_consultation_service


def make_service(tmp_path):
    service = create_consultation_service(tmp_path / "travel.db")
    consultation = service.create_consultation("c1", "客户", "去杭州")
    return service, consultation


def set_requirements(service, consultation, **overrides):
    requirements = {
        "origin": "北京",
        "destination": "杭州",
        "start_date": "2026-10-01",
        "end_date": "2026-10-03",
        "traveler_count": 2,
        "budget": 8000,
    }
    requirements.update(overrides)
    service.update_requirements(consultation.id, requirements)
    return requirements


def tool_message(name, payload, call_id="call-1"):
    return ToolMessage(content=json.dumps(payload), tool_call_id=call_id, name=name)


def search_results(transport_price=1800.0, nightly=680.0, ticket=75.0):
    return [
        tool_message("search_transport_options", {"options": [
            {"option_id": "train-1", "total_price": transport_price},
        ]}),
        tool_message("search_hotel_options", {"options": [
            {"option_id": "hotel-1", "price_per_night": nightly},
        ]}),
        tool_message("search_attractions", {"options": [
            {"option_id": "attr-1", "ticket_price": ticket},
        ]}),
    ]


def test_version_is_derived_from_tool_results(tmp_path):
    service, consultation = make_service(tmp_path)
    set_requirements(service, consultation)

    version = service._record_itinerary_version(
        service.get_consultation(consultation.id), search_results()
    )

    assert version["version_number"] == 1
    assert version["payload"]["transport_option_id"] == "train-1"
    assert version["payload"]["hotel_option_id"] == "hotel-1"
    assert version["payload"]["attraction_option_ids"] == ["attr-1"]


def test_version_budget_scales_tickets_by_traveler_count(tmp_path):
    """门票是每人一张，必须乘以出行人数；交通与酒店已经隐含了人数/晚数。"""
    service, consultation = make_service(tmp_path)
    set_requirements(service, consultation, traveler_count=2)

    version = service._record_itinerary_version(
        service.get_consultation(consultation.id), search_results()
    )
    budget = version["payload"]["budget"]

    assert budget["transport_total"] == 1800.0
    assert budget["hotel_total"] == 680.0 * 2  # 2 晚
    assert budget["attractions_total"] == 75.0 * 2  # 2 人
    assert budget["total"] == 3310.0
    assert budget["within_budget"] is True


def test_version_budget_omitted_when_dates_missing(tmp_path):
    """宁可没有总额，也不要编造一个——版本会被用于审批金额核对。"""
    service, consultation = make_service(tmp_path)
    service.update_requirements(
        consultation.id, {"origin": "北京", "traveler_count": 2}
    )

    version = service._record_itinerary_version(
        service.get_consultation(consultation.id), search_results()
    )

    assert "budget" not in version["payload"]
    assert version["payload"]["transport_option_id"] == "train-1"


def test_no_version_without_transport_and_hotel(tmp_path):
    service, consultation = make_service(tmp_path)
    set_requirements(service, consultation)

    only_attractions = [
        tool_message("search_attractions", {"options": [{"option_id": "attr-1"}]})
    ]
    assert service._record_itinerary_version(
        service.get_consultation(consultation.id), only_attractions
    ) is None


def test_version_ignores_unsuccessful_or_malformed_tool_output(tmp_path):
    service, consultation = make_service(tmp_path)
    set_requirements(service, consultation)

    messages = search_results() + [
        # 工具报错时内容不是 JSON，必须被跳过而不是让整轮崩溃
        ToolMessage(content="Error invoking tool", tool_call_id="c2", name="search_hotel_options"),
        tool_message("search_transport_options", {"options": []}, call_id="c3"),
    ]

    version = service._record_itinerary_version(
        service.get_consultation(consultation.id), messages
    )

    assert version is not None
    assert version["payload"]["transport_option_id"] == "train-1"
    assert "search_transport_options" in version["payload"]["observations"]


def test_each_chat_turn_records_a_new_version(tmp_path):
    service, consultation = make_service(tmp_path)
    set_requirements(service, consultation)

    first = service._record_itinerary_version(
        service.get_consultation(consultation.id), search_results()
    )
    second = service._record_itinerary_version(
        service.get_consultation(consultation.id),
        search_results(transport_price=1500.0, nightly=420.0),
    )

    assert (first["version_number"], second["version_number"]) == (1, 2)
    assert len(service.repository.list_itinerary_versions(consultation.id)) == 2


def test_version_records_observations_for_audit(tmp_path):
    service, consultation = make_service(tmp_path)
    set_requirements(service, consultation)

    version = service._record_itinerary_version(
        service.get_consultation(consultation.id), search_results()
    )

    observations = version["payload"]["observations"]
    assert set(observations) == {
        "search_transport_options",
        "search_hotel_options",
        "search_attractions",
    }

    events = [
        event.event_type
        for event in service.repository.list_audit_events(consultation.id)
    ]
    assert "ITINERARY_VERSION_RECORDED" in events


def test_recorded_version_can_be_selected_for_booking(tmp_path):
    """HITL 预订要求已选定版本，所以这条链路必须端到端可用。"""
    service, consultation = make_service(tmp_path)
    set_requirements(service, consultation)
    version = service._record_itinerary_version(
        service.get_consultation(consultation.id), search_results()
    )

    selected = service.select_itinerary(consultation.id, version["version_id"])

    assert selected["version_id"] == version["version_id"]
    assert selected["version_number"] == 1
