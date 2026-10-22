"""把检查项应用到一次评测运行的结果上。

设计要点：**断言只看结构化的运行记录，不看模型的自然语言**。
"模型说它在预算内"和"版本里 within_budget 真的是 true"是两件事，
后者才是可以自动判断的事实。这也让失败原因可以直接定位到字段。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .tasks import HARD, Check


@dataclass(frozen=True)
class CheckOutcome:
    """The result of applying one check."""

    kind: str
    name: str
    passed: bool
    actual: Any = None
    expected: Any = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "kind": self.kind,
            "name": self.name,
            "passed": self.passed,
            "actual": self.actual,
            "expected": self.expected,
        }
        if self.detail:
            payload["detail"] = self.detail
        return payload


def resolve_path(observation: dict[str, Any], path: str) -> Any:
    """Read a dotted path out of the run observation.

    Missing keys resolve to ``None`` instead of raising: a missing field is a
    normal evaluation result ("the version had no budget"), not a crash.
    """
    if not path:
        return observation
    current: Any = observation
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if current is None:
            return None
    return current


def _compare(op: str, actual: Any, expected: Any) -> tuple[bool, str]:
    if op == "present":
        return actual is not None, ""
    if op == "absent":
        return actual is None, ""
    if op == "truthy":
        return bool(actual), ""
    if op == "falsy":
        return not bool(actual), ""
    if op == "equals":
        return actual == expected, ""
    if op == "contains":
        if actual is None:
            return False, "value is None"
        return expected in actual, ""
    if op == "gte":
        try:
            return float(actual) >= float(expected), ""
        except (TypeError, ValueError):
            return False, f"not comparable: {actual!r} >= {expected!r}"
    if op == "lte":
        try:
            return float(actual) <= float(expected), ""
        except (TypeError, ValueError):
            return False, f"not comparable: {actual!r} <= {expected!r}"
    if op == "within":
        # expected 是 [low, high]，闭区间。
        try:
            low, high = expected
            return float(low) <= float(actual) <= float(high), ""
        except (TypeError, ValueError):
            return False, f"not comparable: {actual!r} in {expected!r}"
    raise ValueError(f"unknown check op: {op}")


def apply_check(check: Check, observation: dict[str, Any]) -> CheckOutcome:
    """Evaluate one check against a run observation."""
    actual = resolve_path(observation, check.path)
    try:
        passed, detail = _compare(check.op, actual, check.value)
    except ValueError as error:
        return CheckOutcome(
            kind=check.kind,
            name=check.name,
            passed=False,
            actual=actual,
            expected=check.value,
            detail=str(error),
        )
    return CheckOutcome(
        kind=check.kind,
        name=check.name,
        passed=passed,
        actual=actual,
        expected=check.value,
        detail=detail,
    )


def apply_checks(
    checks: tuple[Check, ...], observation: dict[str, Any]
) -> list[CheckOutcome]:
    return [apply_check(check, observation) for check in checks]


def hard_failures(outcomes: list[CheckOutcome]) -> list[CheckOutcome]:
    """Hard checks that failed — these decide whether a task passed."""
    return [o for o in outcomes if o.kind == HARD and not o.passed]


def soft_score(outcomes: list[CheckOutcome]) -> float | None:
    """Fraction of soft checks that passed, or ``None`` when there are none."""
    soft = [o for o in outcomes if o.kind != HARD]
    if not soft:
        return None
    return sum(1 for o in soft if o.passed) / len(soft)
