from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .providers import (
    AttractionOption,
    AttractionSearchResult,
    HotelOption,
    HotelSearchResult,
    TransportMode,
    TransportOption,
    TransportSearchResult,
)


def _parse_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", str(value))
    return float(match.group(0).replace(",", "")) if match else None


def parse_hotel_response(payload: dict[str, Any], *, city: str) -> HotelSearchResult:
    items = (payload.get("data") or {}).get("itemList") or []
    options: list[HotelOption] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("shId") or not item.get("name"):
            continue
        options.append(
            HotelOption(
                provider="flyai_fliggy",
                option_id=f"flyai:{item['shId']}",
                hotel_id=str(item["shId"]),
                name=str(item["name"]),
                city=city,
                address=item.get("address"),
                brand_name=item.get("brandName"),
                hotel_type=item.get("star"),
                price_per_night=_parse_number(item.get("price")),
                currency="CNY" if item.get("price") else None,
                rating=_parse_number(item.get("rate")),
                available_rooms=None,
                image_url=item.get("mainPic"),
                detail_url=item.get("detailUrl"),
                latitude=_parse_number(item.get("latitude")),
                longitude=_parse_number(item.get("longitude")),
                nearby_poi=item.get("interestsPoi"),
            )
        )
    return HotelSearchResult(provider="flyai_fliggy", options=options)


def _parse_duration(value: Any) -> int | None:
    number = _parse_number(value)
    return int(number) if number is not None else None


def parse_transport_response(
    payload: dict[str, Any], *, mode: TransportMode
) -> TransportSearchResult:
    items = (payload.get("data") or {}).get("itemList") or []
    options: list[TransportOption] = []
    for item in items:
        journeys = item.get("journeys") or []
        segments = (journeys[0].get("segments") or []) if journeys else []
        segment = segments[0] if segments else None
        if not isinstance(segment, dict):
            continue
        try:
            departure = datetime.fromisoformat(str(segment["depDateTime"]))
            arrival = datetime.fromisoformat(str(segment["arrDateTime"]))
        except (KeyError, TypeError, ValueError):
            continue
        number = segment.get("marketingTransportNo") or "unknown"
        vehicle_type = (
            "high_speed_train"
            if mode == TransportMode.TRAIN
            and str(number).upper().startswith(("G", "D", "C"))
            else mode.value
        )
        options.append(
            TransportOption(
                provider="flyai_fliggy",
                option_id=f"flyai:{mode.value}:{number}:{departure.isoformat()}",
                mode=mode,
                origin=str(segment.get("depCityName", "")),
                destination=str(segment.get("arrCityName", "")),
                departure_date=departure.date(),
                departure_time=departure.time(),
                arrival_time=arrival.time(),
                duration_minutes=_parse_duration(segment.get("duration"))
                or int((arrival - departure).total_seconds() / 60),
                total_price=_parse_number(item.get("adultPrice")) or 0,
                vehicle_type=vehicle_type,
                carrier=segment.get("marketingTransportName"),
                service_number=str(number),
                seat_class=segment.get("seatClassName"),
                booking_url=item.get("jumpUrl"),
            )
        )
    return TransportSearchResult(provider="flyai_fliggy", options=options)


def parse_attraction_response(
    payload: dict[str, Any], *, city: str, visit_date: Any
) -> AttractionSearchResult:
    items = (payload.get("data") or {}).get("itemList") or []
    options: list[AttractionOption] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("id") or not item.get("name"):
            continue
        ticket_info = item.get("ticketInfo") or {}
        raw_price = ticket_info.get("price")
        options.append(
            AttractionOption(
                provider="flyai_fliggy",
                option_id=f"flyai:poi:{item['id']}",
                name=str(item["name"]),
                city=city,
                visit_date=visit_date,
                address=item.get("address"),
                duration_minutes=None,
                ticket_price=_parse_number(raw_price),
                currency="CNY" if raw_price not in (None, "") else None,
                ticket_name=ticket_info.get("ticketName"),
                free_status=item.get("freePoiStatus"),
                image_url=item.get("mainPic"),
                booking_url=item.get("jumpUrl"),
            )
        )
    return AttractionSearchResult(provider="flyai_fliggy", options=options)
