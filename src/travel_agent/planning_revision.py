import re


def parse_planning_revision(message: str) -> dict[str, str | float]:
    constraints: dict[str, str | float] = {}
    budget = re.search(r"(?:预算|预算上限)[^\d]{0,8}(\d+(?:\.\d+)?)\s*(?:元|块)?", message)
    if budget:
        constraints["budget"] = float(budget.group(1))
    if "高铁" in message:
        constraints["transport_mode"] = "train"
    elif "飞机" in message or "航班" in message:
        constraints["transport_mode"] = "flight"
    if "评分更高" in message or "高评分" in message or "评分最高" in message:
        constraints["hotel_preference"] = "highest_rating"
    elif "便宜" in message or "低价" in message:
        constraints["hotel_preference"] = "lowest_price"
    return constraints
