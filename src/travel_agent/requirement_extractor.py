import re
from datetime import date

DATE_PATTERN = r"\d{4}-\d{2}-\d{2}"
# 顾问常用简写：10.17 / 10/17 / 10月17日 / 10-17，都省略了年份。
SHORT_DATE_PATTERN = r"(?<!\d)(\d{1,2})\s*[.\-/月]\s*(\d{1,2})\s*日?(?!\d)"
# 城市名后常紧跟“旅游/玩/给我…”等词，贪婪匹配会把它们一起吃进来。
# 用“这些词之前都算城市名”的方式截断，而不是靠固定长度。
CITY_STOP_WORDS = (
    "旅游", "旅行", "游玩", "玩", "出差", "给我", "帮我", "写", "做", "规划",
    "计划", "的", "和", "跟", "去", "到", "出发", "回来", "玩几天",
)
CITY_PATTERN = r"[\u4e00-\u9fff]{2,12}"


def _normalize_short_dates(message: str) -> list[str]:
    """补全年份并把简写日期转成 ISO。

    年份取“不早于今天”的最近一次：交通查询只接受今天及以后，所以顾问说
    「10.17」时用今年的 10-17；若今年那天已过，则用明年。
    """
    today = date.today()
    normalized: list[str] = []
    for month, day in re.findall(SHORT_DATE_PATTERN, message):
        try:
            candidate = date(today.year, int(month), int(day))
        except ValueError:
            continue
        if candidate < today:
            try:
                candidate = date(today.year + 1, int(month), int(day))
            except ValueError:
                continue
        normalized.append(candidate.isoformat())
    return normalized


def _trim_city(value: str) -> str:
    """Cut trailing filler words the greedy city pattern swallowed."""
    cut = len(value)
    for word in CITY_STOP_WORDS:
        index = value.find(word)
        if index != -1:
            cut = min(cut, index)
    trimmed = value[:cut].strip()
    return trimmed if len(trimmed) >= 2 else value


def extract_requirements(message: str) -> dict[str, str | int | float]:
    """Extract only explicit requirement values from a consultant message."""
    extracted: dict[str, str | int | float] = {}

    origin = re.search(
        rf"从[\s]*({CITY_PATTERN}?)(?:出发|到|，|,|$)|出发地(?:是|为)?[\s]*({CITY_PATTERN})",
        message,
    )
    if origin:
        extracted["origin"] = _trim_city(origin.group(1) or origin.group(2))

    destination = re.search(rf"(?:目的地(?:是|为)?|去)[\s]*({CITY_PATTERN})", message)
    if destination:
        extracted["destination"] = _trim_city(destination.group(1))

    # 完整 ISO 日期优先；没有时再退回简写，避免两种写法混用时顺序错乱。
    dates = re.findall(DATE_PATTERN, message) or _normalize_short_dates(message)
    if dates:
        extracted["start_date"] = dates[0]
    if len(dates) > 1:
        extracted["end_date"] = dates[1]

    travelers = re.search(r"(\d+)\s*(?:人|位)", message)
    if travelers:
        extracted["traveler_count"] = int(travelers.group(1))

    budget = re.search(r"(?:预算|费用|花费)[^\d]{0,8}(\d+(?:\.\d+)?)\s*(?:元|块)?", message)
    if budget:
        extracted["budget"] = float(budget.group(1))

    return extracted
