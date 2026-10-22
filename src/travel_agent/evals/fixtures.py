"""评测专用的数据准备。

**为什么评测要单独 seed 数据**：随包的 sqlite 样本把交通固定在一个日期上，
那个日期一旦过去，评测任务就会永远拿不到交通选项——而"没有交通选项"会让
版本无法生成，于是所有任务的硬断言全部失败，看起来像是模型变差了。

评测需要的是**不受当前日期影响**的数据：总是在"未来某天"有票。因此这里
提供 `prepare_eval_database()`，在临时库里按评测实际要用的日期写数据。
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from ..sqlite_providers import initialize_travel_data


def eval_trip_date(offset_days: int = 14) -> date:
    """A trip date far enough ahead to look like a real future booking.

    Being relative to ``today`` keeps the suite meaningful whenever it runs;
    a hard-coded date silently rots.
    """
    return date.today() + timedelta(days=offset_days)


def prepare_eval_database(database: str | Path, *, trip_date: date | None = None) -> date:
    """Seed a database whose transport covers ``trip_date``.

    Returns the seeded date so tasks can phrase their prompts around it.
    """
    seeded = trip_date or eval_trip_date()
    initialize_travel_data(database, transport_date=seeded.isoformat())
    return seeded


def prompt_for_date(trip_date: date) -> str:
    """Render a date the way a consultant would write it."""
    return f"{trip_date.year}年{trip_date.month}月{trip_date.day}日"


def check_eval_environment() -> list[str]:
    """Reject environments that would fail every task for the same reason.

    An evaluation that cannot reach the model at all is not a measurement: it
    produces a full page of task failures that reads exactly like a model
    regression. Catching the few known causes up front is far cheaper than
    explaining a 0/15 report afterwards.
    """
    import importlib.util
    import os

    problems: list[str] = []

    # httpx 走 SOCKS 代理时必须有 socksio，缺了每个任务都会 ImportError。
    proxy = (
        os.getenv("all_proxy")
        or os.getenv("ALL_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("HTTPS_PROXY")
        or ""
    )
    if proxy.startswith("socks") and importlib.util.find_spec("socksio") is None:
        problems.append(
            f"proxy {proxy!r} needs the 'socksio' package "
            "(pip install 'httpx[socks]'), or unset all_proxy/ALL_PROXY"
        )

    return problems
