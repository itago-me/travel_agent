"""需求抽取的日期解析：显式年份与相对日期。

两个缺陷都是在策划阶段 7（评测体系）时核查发现的，属于"单元测试抓不到、
人工走查也难碰上"的类型，因此修复后必须固化成用例，防止复发。

缺陷 1：显式年份被静默顺延
    今天是 2026-10-20 时，输入「2026年10月1日」被解析成 2027-10-01。
    根因是 _normalize_short_dates 无条件执行"不早于今天"的推断，没有区分
    用户是否明确给出了年份。顾问查历史/复盘时会得到完全错误的方案日期，
    而且不会有任何报错。

缺陷 2：相对日期完全抽不出
    「下周三出发」抽出的 requirements 是空字典。顾问口语里这类表述很常见，
    丢弃后模型只能从原始消息里猜，猜错概率很高。

约定：以下用例都基于固定的"今天"，避免测试在不同日期跑出不同结果。
"""

import pytest

from travel_agent.requirement_extractor import (
    _normalize_short_dates,
    _relative_dates,
    extract_requirements,
)


@pytest.fixture
def fixed_today(monkeypatch):
    """把"今天"固定为 2026-10-20（周二），使周边计算可预期。"""
    import travel_agent.requirement_extractor as module

    class _FixedDate(module.date):
        @classmethod
        def today(cls):
            return cls(2026, 10, 20)

    monkeypatch.setattr(module, "date", _FixedDate)
    return _FixedDate(2026, 10, 20)


# --- 缺陷 1：显式年份 -------------------------------------------------------


def test_explicit_year_is_respected_even_when_in_the_past(fixed_today):
    """顾问写 2026 年就是在说 2026 年，不能因为已过就顺延到 2027。"""
    assert _normalize_short_dates("2026年10月1日") == ["2026-10-01"]


def test_explicit_year_does_not_leak_into_the_next_day(fixed_today):
    """区间两端的年份要一致：10月3日 跟随前一个显式年份，不能变成明年。"""
    assert _normalize_short_dates("2026年10月1日到10月3日") == [
        "2026-10-01",
        "2026-10-03",
    ]


def test_yearless_date_still_rolls_forward(fixed_today):
    """省略年份时仍取"不早于今天"的最近一次，这条原有行为不能回退。"""
    assert _normalize_short_dates("10月1日") == ["2027-10-01"]


def test_yearless_date_in_the_future_stays_this_year(fixed_today):
    assert _normalize_short_dates("12月25日") == ["2026-12-25"]


def test_impossible_explicit_date_is_skipped(fixed_today):
    assert _normalize_short_dates("2026年2月30日") == []


def test_iso_dates_are_untouched(fixed_today):
    assert extract_requirements("2026-10-01到2026-10-03") == {
        "start_date": "2026-10-01",
        "end_date": "2026-10-03",
    }


# --- 缺陷 2：相对日期 -------------------------------------------------------


def test_tomorrow_and_day_after(fixed_today):
    assert _relative_dates("明天出发") == ["2026-10-21"]
    assert _relative_dates("后天出发") == ["2026-10-22"]
    assert _relative_dates("大后天出发") == ["2026-10-23"]


def test_today_counts_as_today(fixed_today):
    assert _relative_dates("今天就走") == ["2026-10-20"]


def test_this_week_weekday(fixed_today):
    """今天周二，本周五是 10-23。"""
    assert _relative_dates("本周五") == ["2026-10-23"]


def test_bare_weekday_means_nearest_upcoming(fixed_today):
    """只说「周五」时取最近的那个周五，即本周。"""
    assert _relative_dates("周五去杭州") == ["2026-10-23"]


def test_next_week_weekday(fixed_today):
    """今天周二，「下周三」是 10-28（下周）。"""
    assert _relative_dates("下周三出发") == ["2026-10-28"]


def test_week_after_next(fixed_today):
    """「下下周一」= 11-02，不能多算一周。"""
    assert _relative_dates("下下周一") == ["2026-11-02"]


def test_bare_weekday_on_the_same_day_means_next_occurrence(fixed_today):
    """今天就是周二时说「周二」，指下一次（下周），不是今天。"""
    assert _relative_dates("周二") == ["2026-10-27"]


def test_relative_date_produces_a_start_date(fixed_today):
    result = extract_requirements("下周三去杭州，2人")
    assert result["start_date"] == "2026-10-28"
    assert result["destination"] == "杭州"
    assert result["traveler_count"] == 2


# --- 歧义表述宁可不猜 -------------------------------------------------------


@pytest.mark.parametrize("message", ["下个月去杭州", "过阵子去杭州", "国庆去杭州"])
def test_ambiguous_expressions_are_not_guessed(fixed_today, message):
    """歧义大的说法不推断：猜错比不抽更糟，会让后续规划建立在错误日期上。"""
    assert "start_date" not in extract_requirements(message)


# --- 优先级：明确日期优先于相对表述 -----------------------------------------


def test_explicit_date_wins_over_relative(fixed_today):
    """简写日期（10月28日）优先于相对解析，且不被其覆盖。"""
    result = extract_requirements("10月28日出发")
    assert result["start_date"] == "2026-10-28"


# --- 出行人数：中文表述 -----------------------------------------------------


@pytest.mark.parametrize(
    "message,expected",
    [
        ("我一个人去杭州", 1),
        ("单人出行", 1),
        ("就我自己去", 1),
        ("两个人去杭州", 2),
        ("我和老婆两个人", 2),
        ("我们俩", 2),
        ("一家三口", 3),
        ("我们仨", 3),
        ("2人", 2),
        ("三位", 3),
    ],
)
def test_chinese_traveler_counts_are_recognized(fixed_today, message, expected):
    """只认阿拉伯数字会漏掉顾问最常用的说法，而人数直接影响预算合计。"""
    assert extract_requirements(message)["traveler_count"] == expected


@pytest.mark.parametrize("message", ["我们一家人去杭州", "一起去杭州", "和朋友去杭州"])
def test_unknown_party_size_is_not_guessed(fixed_today, message):
    """人数不明时不猜：猜错会让交通票价和门票合计失真，比不填更糟。"""
    assert "traveler_count" not in extract_requirements(message)


# --- 预算金额单位 -----------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected",
    [
        ("预算2万元以内", 20_000.0),
        ("预算1.5万", 15_000.0),
        ("预算8000元", 8_000.0),
        ("预算3千", 3_000.0),
        ("预算500", 500.0),
        ("预算大概2万元左右", 20_000.0),
    ],
)
def test_budget_units_are_converted(fixed_today, message, expected):
    """把"2万元"读成 2 会让预算判断彻底失真——充裕的预算被判成超支。"""
    assert extract_requirements(message)["budget"] == expected
