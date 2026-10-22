"""把一次评测报告整理成可读的基线。

`eval` 命令负责跑并落盘 JSON；这个工具负责把 JSON 变成"人一眼能看懂、
能拿来做决策"的基线，回答 7.2 真正关心的问题：

- 每个任务的通过率是多少（不是单次结果）；
- 哪些任务不稳定（flaky），哪些稳定失败；
- 成本：工具调用次数、耗时——预算再充足也不该放任模型乱查。

用法：
    python -m travel_agent.evals.baseline evals/results/<file>.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text("utf-8"))


def _group(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for task in report.get("tasks", []):
        grouped.setdefault(task["task_id"], []).append(task)
    return grouped


def build_baseline(report: dict[str, Any]) -> dict[str, Any]:
    """Condense a report into per-task rates plus cost aggregates."""
    grouped = _group(report)
    rows = []
    for task_id, results in grouped.items():
        passes = sum(1 for r in results if r.get("passed"))
        tool_counts = [
            r["observation"].get("tool_call_count", 0) for r in results
        ]
        durations = [r.get("duration_seconds", 0) for r in results]
        soft_scores = [
            r.get("soft_score") for r in results if r.get("soft_score") is not None
        ]
        rows.append(
            {
                "task_id": task_id,
                "runs": len(results),
                "passes": passes,
                "pass_rate": round(passes / len(results), 3),
                "flaky": 0 < passes < len(results),
                "avg_tool_calls": round(statistics.mean(tool_counts), 2) if tool_counts else 0,
                "max_tool_calls": max(tool_counts) if tool_counts else 0,
                "avg_seconds": round(statistics.mean(durations), 1) if durations else 0,
                "avg_soft_score": (
                    round(statistics.mean(soft_scores), 2) if soft_scores else None
                ),
            }
        )
    rows.sort(key=lambda row: (row["pass_rate"], row["task_id"]))
    return {
        "model": report.get("model"),
        "repeat": report.get("stability", {}).get("repeat", 1),
        "tasks": rows,
    }


def format_baseline(baseline: dict[str, Any]) -> str:
    """Render the baseline as a fixed-width table."""
    lines = [
        f"model={baseline['model']} repeat={baseline['repeat']}",
        "",
        f"{'task':32} {'pass':>7} {'tools(avg/max)':>15} {'sec':>6} {'soft':>5}",
        "-" * 72,
    ]
    for row in baseline["tasks"]:
        rate = f"{row['passes']}/{row['runs']}"
        if row["flaky"]:
            rate += " !"
        soft = row["avg_soft_score"]
        lines.append(
            f"{row['task_id']:32} {rate:>7} "
            f"{row['avg_tool_calls']:>7}/{row['max_tool_calls']:<7} "
            f"{row['avg_seconds']:>6} {('-' if soft is None else soft):>5}"
        )
    lines.append("-" * 72)
    flaky = [row["task_id"] for row in baseline["tasks"] if row["flaky"]]
    failed = [row["task_id"] for row in baseline["tasks"] if row["pass_rate"] == 0]
    lines.append(f"flaky ({len(flaky)}): {', '.join(flaky) or 'none'}")
    lines.append(f"always_fail ({len(failed)}): {', '.join(failed) or 'none'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    arguments = argv if argv is not None else sys.argv[1:]
    if not arguments:
        print("usage: python -m travel_agent.evals.baseline <report.json>")
        return 2
    baseline = build_baseline(_load(arguments[0]))
    print(format_baseline(baseline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def compare_reports(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    shared_only: bool = False,
) -> dict[str, Any]:
    """Diff two reports so a regression is visible without reading raw JSON.

    The comparison is per task id, because that is the unit a person reasons
    about ("happy-path broke") — not the aggregate pass count, which can stay
    flat while individual tasks swap places.
    """
    before = build_baseline(previous)["tasks"]
    after = build_baseline(current)["tasks"]
    before_by_id = {row["task_id"]: row for row in before}
    after_by_id = {row["task_id"]: row for row in after}

    regressed: list[str] = []
    fixed: list[str] = []
    added: list[str] = []
    removed: list[str] = []
    tool_changes: list[dict[str, Any]] = []

    for task_id, row in after_by_id.items():
        if task_id not in before_by_id:
            added.append(task_id)
            continue
        old = before_by_id[task_id]
        if row["pass_rate"] < old["pass_rate"]:
            regressed.append(task_id)
        elif row["pass_rate"] > old["pass_rate"]:
            fixed.append(task_id)
        delta = row["avg_tool_calls"] - old["avg_tool_calls"]
        if abs(delta) >= 1:
            tool_changes.append(
                {
                    "task_id": task_id,
                    "before": old["avg_tool_calls"],
                    "after": row["avg_tool_calls"],
                    "delta": round(delta, 2),
                }
            )

    if not shared_only:
        # 部分运行（--only）时"其余任务消失"没有意义，会被误读成任务被删除。
        removed.extend(
            task_id for task_id in before_by_id if task_id not in after_by_id
        )

    return {
        "regressed": sorted(regressed),
        "fixed": sorted(fixed),
        "added": sorted(added),
        "removed": sorted(removed),
        "tool_changes": sorted(tool_changes, key=lambda item: item["delta"]),
    }


def format_comparison(comparison: dict[str, Any]) -> str:
    lines = []
    if comparison["regressed"]:
        lines.append(f"REGRESSED ({len(comparison['regressed'])}):")
        lines.extend(f"  - {task_id}" for task_id in comparison["regressed"])
    if comparison["fixed"]:
        lines.append(f"fixed ({len(comparison['fixed'])}):")
        lines.extend(f"  + {task_id}" for task_id in comparison["fixed"])
    if comparison["added"]:
        lines.append(f"added tasks: {', '.join(comparison['added'])}")
    if comparison["removed"]:
        lines.append(f"removed tasks: {', '.join(comparison['removed'])}")
    if comparison["tool_changes"]:
        lines.append("tool call changes (per task, avg):")
        for change in comparison["tool_changes"]:
            lines.append(
                f"  {change['task_id']:32} "
                f"{change['before']} -> {change['after']} "
                f"({change['delta']:+.2f})"
            )
    if not lines:
        lines.append("no differences")
    return "\n".join(lines)
