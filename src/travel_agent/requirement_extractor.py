import re


DATE_PATTERN = r"\d{4}-\d{2}-\d{2}"
CITY_PATTERN = r"[\u4e00-\u9fff]{2,12}"


def extract_requirements(message: str) -> dict[str, str | int | float]:
    """Extract only explicit requirement values from a consultant message."""
    extracted: dict[str, str | int | float] = {}

    origin = re.search(
        rf"从[\s]*({CITY_PATTERN}?)(?:出发|到|，|,|$)|出发地(?:是|为)?[\s]*({CITY_PATTERN})",
        message,
    )
    if origin:
        extracted["origin"] = origin.group(1) or origin.group(2)

    destination = re.search(rf"(?:目的地(?:是|为)?|去)[\s]*({CITY_PATTERN})", message)
    if destination:
        extracted["destination"] = destination.group(1)

    dates = re.findall(DATE_PATTERN, message)
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
