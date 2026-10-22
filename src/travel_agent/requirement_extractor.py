import re
from datetime import date, timedelta

DATE_PATTERN = r"\d{4}-\d{2}-\d{2}"
# 顾问常用简写：10.17 / 10/17 / 10月17日 / 2026年10月17日。
# 年份是可选的捕获组：用户写了年份就必须尊重它，没写才按"不早于今天"推断。
SHORT_DATE_PATTERN = (
    r"(?:(\d{4})\s*年\s*)?(?<!\d)(\d{1,2})\s*[.\-/月]\s*(\d{1,2})\s*日?(?!\d)"
)
# 城市名后常紧跟“旅游/玩/给我…”等词，贪婪匹配会把它们一起吃进来。
# 用“这些词之前都算城市名”的方式截断，而不是靠固定长度。
CITY_STOP_WORDS = (
    "旅游", "旅行", "游玩", "玩", "出差", "给我", "帮我", "写", "做", "规划",
    "计划", "的", "和", "跟", "去", "到", "出发", "回来", "玩几天",
)
CITY_PATTERN = r"[\u4e00-\u9fff]{2,12}"

# 相对日期：顾问口语里极常见，此前被直接丢弃，导致 requirements 为空而
# 流程照常往下走，模型只能从原始消息里猜日期。
# 只处理有确定含义、不需要额外上下文就能换算的表述；
# "下个月""过阵子""国庆"这类歧义大的不猜，宁可不抽。
_WEEKDAYS = {
    "周一": 0, "周二": 1, "周三": 2, "周四": 3,
    "周五": 4, "周六": 5, "周日": 6, "周天": 6,
    "星期一": 0, "星期二": 1, "星期三": 2, "星期四": 3,
    "星期五": 4, "星期六": 5, "星期日": 6, "星期天": 6,
}
_RELATIVE_DAY_OFFSETS = {"今天": 0, "今日": 0, "明天": 1, "明日": 1, "后天": 2, "大后天": 3}
RELATIVE_DATE_PATTERN = re.compile(
    r"(?P<day>今天|今日|明天|明日|大后天|后天)"
    # 只捕获"下/下下/本/这"，紧跟的"周"属于 weekday 部分。
    r"|(?P<which>下下|下|本|这)?\s*"
    r"(?P<weekday>周[一二三四五六日天]|星期[一二三四五六日天])"
)


def _normalize_short_dates(message: str) -> list[str]:
    """补全年份并把简写日期转成 ISO。

    三种情形都要区别对待：

    - **用户写了年份**（``2026年10月1日``）：照用，不做任何顺延。顾问可能是
      在查历史/复盘，把 ``2026年`` 静默改成 ``2027年`` 会生成完全错误的方案，
      而且不会报错——这类错误只有对着输入核对才能发现。
    - **省略年份但前面已有显式年份**（``2026年10月1日到10月3日``）：跟随后者。
      区间两端属于同一次旅行，把结束日期推到明年会得到一个跨年的荒谬区间。
    - **全文都省略年份**（``10.17``）：取"不早于今天"的最近一次。交通查询只
      接受今天及以后，所以今年那天已过时用明年。
    """
    today = date.today()
    normalized: list[str] = []
    anchor_year: int | None = None
    for year, month, day in re.findall(SHORT_DATE_PATTERN, message):
        month_num, day_num = int(month), int(day)
        if year:
            explicit = _build_date(int(year), month_num, day_num)
            if explicit is None:
                continue
            # 显式年份同时成为后续省略年份日期的锚点。
            anchor_year = explicit.year
            normalized.append(explicit.isoformat())
            continue
        if anchor_year is not None:
            anchored = _build_date(anchor_year, month_num, day_num)
            if anchored is not None:
                normalized.append(anchored.isoformat())
            continue
        inferred = _most_recent_future(today, month_num, day_num)
        if inferred is not None:
            normalized.append(inferred.isoformat())
    return normalized


def _build_date(year: int, month: int, day: int) -> date | None:
    """Build a date, or ``None`` when the combination is not a real date."""
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _most_recent_future(today: date, month: int, day: int) -> date | None:
    """The nearest date not earlier than ``today`` for a yearless month/day."""
    candidate = _build_date(today.year, month, day)
    if candidate is None:
        return None
    if candidate < today:
        candidate = _build_date(today.year + 1, month, day)
    return candidate


def _relative_dates(message: str) -> list[str]:
    """把"明天/后天/本周五/下周三"等口语表述换算成 ISO 日期。

    歧义太大的表述（"下个月""过阵子""国庆"）一律不猜——猜错比不抽更糟，
    因为它会让后续所有规划建立在错误的日期上。抽不到时由上层明确告知用户
    补充信息，而不是静默留空。

    周的算法与中文习惯一致：一周从周一开始，``下周X`` 指下一周的周 X。
    """
    today = date.today()
    found: list[str] = []
    for match in RELATIVE_DATE_PATTERN.finditer(message):
        day_word = match.group("day")
        if day_word:
            found.append(
                (today + timedelta(days=_RELATIVE_DAY_OFFSETS[day_word])).isoformat()
            )
            continue
        weekday_word = match.group("weekday")
        if weekday_word not in _WEEKDAYS:
            continue
        target = _WEEKDAYS[weekday_word]
        prefix = match.group("which") or ""
        # 从"本周一"起算最清楚：本周=+0，下周=+1，下下周=+2 周，再加周内偏移。
        # 若以"距今天的天数"为基准再加周数，会多算一周。
        monday = today - timedelta(days=today.weekday())
        week_offset = {"下": 1, "下下": 2}.get(prefix, 0)
        if not prefix:
            # 只说"周三"且今天就是周三时，指下一次（下周），不是今天。
            if target == today.weekday():
                week_offset = 1
        elif prefix in ("本", "这") and target < today.weekday():
            # "本周一"在今天是周二时已经过去，仍按字面给本周，交由上层校验。
            week_offset = 0
        found.append(
            (monday + timedelta(weeks=week_offset, days=target)).isoformat()
        )
    return found


# 中文数字 → 阿拉伯数字。只覆盖旅行场景里真实会出现的范围：
# "一家三口""我和老婆两个人""我们仨"。
_CN_DIGITS = {
    "一": 1, "两": 2, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}
_CN_SOLO = ("我一个人", "我自己一个", "单人", "独自", "就我自己", "一个人去", "自己一个")


def _chinese_traveler_count(message: str) -> int | None:
    """从中文表述里推断出行人数。

    只处理含义明确、不需要额外上下文的说法；"我们""一家人"这类不知人数的
    表述一律不猜——猜错会让预算合计失真，比不填更糟。
    """
    # "一个人"/"单人" 是最常见的单人表述，且优先级最高：
    # "我一个人" 里没有任何数字，但含义明确。
    if any(word in message for word in _CN_SOLO):
        return 1
    # "N口"（一家三口）、"N人/位"（两人、三位）
    # "两个人" 里数字和量词之间有个"个"，要一并允许。
    match = re.search(r"([一两二三四五六七八九十])\s*个?\s*(?:口|人|位)", message)
    if match:
        return _CN_DIGITS.get(match.group(1))
    # "我们仨" / "我们俩"
    if "仨" in message:
        return 3
    if "俩" in message:
        return 2
    return None


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

    # 完整 ISO 日期优先，其次简写日期，最后才是相对日期（"下周三"）。
    # 三者都可能出现，按确定性从高到低取值，避免相对表述覆盖明确日期。
    dates = (
        re.findall(DATE_PATTERN, message)
        or _normalize_short_dates(message)
        or _relative_dates(message)
    )
    if dates:
        extracted["start_date"] = dates[0]
    if len(dates) > 1:
        extracted["end_date"] = dates[1]

    travelers = re.search(r"(\d+)\s*(?:人|位)", message)
    if travelers:
        extracted["traveler_count"] = int(travelers.group(1))
    else:
        # 顾问口语里极少说"1人"，通常说"我一个人"；"两人"/"一家三口"这类
        # 中文数字也很常见。只认阿拉伯数字会漏掉它们，而人数直接影响交通
        # 票价与景点门票合计——漏了就必然算错预算。
        chinese_count = _chinese_traveler_count(message)
        if chinese_count is not None:
            extracted["traveler_count"] = chinese_count

    # 金额单位必须识别："2万元" 被读成 2 会让预算判断彻底失真
    # （充裕的预算被判成超支）。评测基线把这个缺陷直接抓了出来。
    budget = re.search(
        r"(?:预算|费用|花费)[^\d]{0,8}(\d+(?:\.\d+)?)\s*(万|千|w|k)?\s*(?:元|块)?",
        message,
    )
    if budget:
        amount = float(budget.group(1))
        unit = (budget.group(2) or "").lower()
        if unit == "万" or unit == "w":
            amount *= 10_000
        elif unit == "千" or unit == "k":
            amount *= 1_000
        extracted["budget"] = amount

    return extracted
