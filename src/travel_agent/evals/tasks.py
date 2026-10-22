"""评测任务的定义与加载。

一条任务 = 一段顾问输入 + 一组检查项。任务用目录下的 YAML 描述，保持
"新增用例只需要写一个文件"，不用改代码。

检查项分两类，这个区分是整个评测体系的基石：

- **硬断言（hard）**：必须 100% 通过。失败即视为回归，不因模型波动而豁免。
  只有那些不依赖模型随机性的性质才允许放进这一类（版本是否生成、审计是否
  完整、敏感操作是否触发审批）。
- **软指标（soft）**：只记录分数，不阻断。真实模型有波动，把"是否在预算内"
  这类指标当硬断言会让评测永远不稳定，反而失去意义。

把两者混在一起是评测最常见的失败模式：要么因为波动而无法判定成败，要么
为了稳定而把真正的回归也放过。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HARD = "hard"
SOFT = "soft"


@dataclass(frozen=True)
class Check:
    """One assertion applied to a finished evaluation run."""

    kind: str  # "hard" | "soft"
    name: str
    # 检查类型：present / absent / equals / contains / gte / lte / within
    op: str
    path: str = ""
    value: Any = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in (HARD, SOFT):
            raise ValueError(f"check kind must be '{HARD}' or '{SOFT}': {self.kind}")


@dataclass(frozen=True)
class EvalTask:
    """A single consultant scenario plus the checks it must satisfy."""

    id: str
    message: str
    description: str = ""
    checks: tuple[Check, ...] = ()
    # 任务可以要求先追加若干轮输入；空表示单轮。
    followups: tuple[str, ...] = ()
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def hard_checks(self) -> tuple[Check, ...]:
        return tuple(check for check in self.checks if check.kind == HARD)

    @property
    def soft_checks(self) -> tuple[Check, ...]:
        return tuple(check for check in self.checks if check.kind == SOFT)


def _parse_check(raw: dict[str, Any], task_id: str) -> Check:
    try:
        return Check(
            kind=raw["kind"],
            name=raw["name"],
            op=raw["op"],
            path=raw.get("path", ""),
            value=raw.get("value"),
            description=raw.get("description", ""),
        )
    except KeyError as error:
        raise ValueError(f"task {task_id}: check missing field {error}") from error


def parse_task(raw: dict[str, Any]) -> EvalTask:
    """Build one task from its decoded definition."""
    try:
        task_id = raw["id"]
        message = raw["message"]
    except KeyError as error:
        raise ValueError(f"task definition missing field {error}") from error
    checks = tuple(_parse_check(item, task_id) for item in raw.get("checks", ()))
    return EvalTask(
        id=task_id,
        message=message,
        description=raw.get("description", ""),
        checks=checks,
        followups=tuple(raw.get("followups", ())),
        tags=tuple(raw.get("tags", ())),
    )


def load_tasks(directory: str | Path) -> list[EvalTask]:
    """Load every task file in a directory.

    ``.json`` files are read directly. ``.yaml`` requires PyYAML; when it is not
    installed the file is skipped with a clear error rather than silently
    dropping tasks — a silently-skipped task looks like a passing one.
    """
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"eval task directory not found: {path}")

    tasks: list[EvalTask] = []
    for file in sorted(path.glob("*.json")):
        data = json.loads(file.read_text("utf-8"))
        # 两种写法都接受：裸数组，或 {"tasks": [...]}。
        # 后者便于在文件里放说明性字段而不污染任务列表。
        if isinstance(data, dict):
            data = data.get("tasks", [])
        tasks.extend(parse_task(item) for item in data)
    for file in sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml")):
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - 取决于环境
            raise RuntimeError(
                f"PyYAML is required to load {file.name}; install pyyaml "
                "or convert the task file to JSON"
            ) from error
        data = yaml.safe_load(file.read_text("utf-8")) or []
        if isinstance(data, dict):
            data = data.get("tasks", [])
        tasks.extend(parse_task(item) for item in data)

    ids = [task.id for task in tasks]
    duplicates = {task_id for task_id in ids if ids.count(task_id) > 1}
    if duplicates:
        raise ValueError(f"duplicate eval task ids: {sorted(duplicates)}")
    return tasks
