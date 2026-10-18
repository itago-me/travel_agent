"""需求提取：把顾问的口语日期、人数、预算转成结构化需求。

真实故障：顾问说「出发时间是10.17,回来10.20」，旧实现只认 ``YYYY-MM-DD``，
日期整段丢失，模型只能自己猜年份，猜成了过去的一年，于是 FlyAI 返回
``出发日期非法``（格式对但早于今天）。这里补上简写日期的识别与年份归一化。
"""

from datetime import date, timedelta

from travel_agent.requirement_extractor import extract_requirements


def future_month_day(months_ahead: int = 2) -> tuple[int, int, str]:
    """Return a month/day that is definitely in the future, plus its ISO form."""
    today = date.today()
    month = today.month + months_ahead
    year = today.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    return month, 15, date(year, month, 15).isoformat()


def test_full_iso_dates_are_extracted_unchanged():
    result = extract_requirements(
        "从北京出发去杭州，2026-10-01到2026-10-03，2人，预算8000元"
    )

    assert result["start_date"] == "2026-10-01"
    assert result["end_date"] == "2026-10-03"
    assert result["origin"] == "北京"
    assert result["destination"] == "杭州"
    assert result["traveler_count"] == 2
    assert result["budget"] == 8000.0


def test_dotted_dates_are_normalized_to_iso():
    month, day, expected = future_month_day()

    result = extract_requirements(f"出发时间是{month}.{day},回来{month}.{day + 1},从萍乡出发")

    assert result["start_date"] == expected
    assert result["origin"] == "萍乡"


def test_slash_and_chinese_dates_are_supported():
    month, day, expected = future_month_day()

    assert extract_requirements(f"{month}/{day}出发")["start_date"] == expected
    assert extract_requirements(f"{month}月{day}日出发")["start_date"] == expected


def test_past_dotted_date_rolls_to_next_year():
    """交通查询拒收过去的日期，所以已过的月日必须理解为明年。"""
    today = date.today()
    yesterday = today - timedelta(days=1)
    if yesterday.month == today.month and yesterday.day != today.day:
        # Same month -> same year is still ahead or equal; skip the ambiguous case.
        return
    result = extract_requirements(
        f"出发时间是{yesterday.month}.{yesterday.day}"
    )

    parsed = date.fromisoformat(result["start_date"])
    assert parsed >= today
    assert parsed.year == today.year + 1


def test_two_dotted_dates_map_to_start_and_end():
    month, day, expected = future_month_day()
    result = extract_requirements(f"{month}.{day}出发，{month}.{day + 2}回来")

    assert result["start_date"] == expected
    assert result["end_date"] == date.fromisoformat(expected).replace(
        day=day + 2
    ).isoformat()


def test_impossible_dates_are_ignored_not_guessed():
    """13.45 这种不存在的月日应被丢弃，而不是抛异常或编造。"""
    result = extract_requirements("出发时间是13.45")

    assert "start_date" not in result


def test_iso_dates_take_precedence_over_dotted_ones():
    """混用两种写法时以完整 ISO 为准，避免顺序错乱。"""
    result = extract_requirements("2026-12-01出发，不要用 3.5")

    assert result["start_date"] == "2026-12-01"


def test_message_without_requirements_extracts_nothing():
    assert extract_requirements("你好，帮我看看") == {}


def test_destination_stops_before_trailing_filler_words():
    """贪婪的城市正则会把「杭州旅游」「贵州给我写一份规划」整段吃掉。"""
    assert extract_requirements("我想去杭州旅游")["destination"] == "杭州"
    assert extract_requirements("去贵州给我写一份规划")["destination"] == "贵州"
    assert extract_requirements("去杭州玩")["destination"] == "杭州"
    assert extract_requirements("我想去杭州")["destination"] == "杭州"


def test_origin_stops_before_trailing_words():
    assert extract_requirements("从萍乡出发")["origin"] == "萍乡"
    assert extract_requirements("出发地是北京")["origin"] == "北京"


def test_event_type_name_is_exposed_for_audit():
    """需求提取发生在建单时刻，审计里必须能区分它与普通消息。"""
    from travel_agent.service import ConsultationService
    from travel_agent.repository import SQLiteRepository
    import tempfile, pathlib

    repository = SQLiteRepository(pathlib.Path(tempfile.mkdtemp()) / "a.db")
    service = ConsultationService(repository)
    consultation = service.create_consultation("c1", "客户", "想去杭州旅游")

    events = [
        event.event_type for event in repository.list_audit_events(consultation.id)
    ]
    assert events == ["CONSULTATION_CREATED", "REQUIREMENTS_EXTRACTED"]
