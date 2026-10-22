"""运行评测任务并产出结构化的观察记录。

每条任务在一个**独立的临时数据库**里跑：任务之间不能共享咨询状态，否则
前一条的需求抽取会污染后一条，评测结果就不可复现了。

观察记录（observation）是断言唯一的输入，它只包含**可以自动核对的事实**：
状态、版本、预算、工具调用、审计事件。不包含模型说了什么漂亮话。
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..application import create_consultation_service
from .fixtures import prepare_eval_database
from .checks import CheckOutcome, apply_checks, hard_failures, soft_score
from .tasks import EvalTask


@dataclass
class TaskResult:
    """One task's outcome."""

    task_id: str
    passed: bool
    observation: dict[str, Any]
    outcomes: list[CheckOutcome] = field(default_factory=list)
    error: str | None = None
    duration_seconds: float = 0.0

    @property
    def soft_score(self) -> float | None:
        return soft_score(self.outcomes)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "passed": self.passed,
            "duration_seconds": round(self.duration_seconds, 3),
            "checks": [outcome.as_dict() for outcome in self.outcomes],
            "observation": self.observation,
        }
        score = self.soft_score
        if score is not None:
            payload["soft_score"] = round(score, 3)
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class EvalReport:
    """The result of a whole eval run."""

    results: list[TaskResult] = field(default_factory=list)
    model: str = ""
    started_at: str = ""
    repeat: int = 1

    def per_task(self) -> dict[str, list[TaskResult]]:
        """Group results by task id, preserving run order."""
        grouped: dict[str, list[TaskResult]] = {}
        for result in self.results:
            grouped.setdefault(result.task_id, []).append(result)
        return grouped

    def stability(self) -> dict[str, Any]:
        """Report how consistently each task passed across repeats.

        This is the part a single run cannot produce. A task that passes 2 of 3
        times is *flaky*, which matters more than a task that fails 3 of 3: the
        consistent failure is a known bug, the flaky one silently erodes trust in
        every later comparison.
        """
        grouped = self.per_task()
        flaky: list[str] = []
        always_fail: list[str] = []
        always_pass: list[str] = []
        for task_id, results in grouped.items():
            passes = sum(1 for result in results if result.passed)
            if passes == len(results):
                always_pass.append(task_id)
            elif passes == 0:
                always_fail.append(task_id)
            else:
                flaky.append(task_id)
        return {
            "repeat": self.repeat,
            "always_pass": always_pass,
            "always_fail": always_fail,
            "flaky": flaky,
        }

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed(self) -> int:
        return len(self.results) - self.passed

    @property
    def soft_average(self) -> float | None:
        scores = [r.soft_score for r in self.results if r.soft_score is not None]
        if not scores:
            return None
        return sum(scores) / len(scores)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "started_at": self.started_at,
            # repeat 同时出现在这里和 stability 里：顶层方便人读，
            # stability 里的是统计时的实际样本数，两者可能因中断而不一致。
            "repeat": self.repeat,
            "summary": {
                "total": len(self.results),
                "passed": self.passed,
                "failed": self.failed,
            },
            "stability": self.stability(),
            "tasks": [result.as_dict() for result in self.results],
        }
        average = self.soft_average
        if average is not None:
            payload["summary"]["soft_average"] = round(average, 3)
        # 按任务聚合的通过率：比逐次罗列更能看出稳定性。
        payload["summary"]["task_pass_rates"] = {
            task_id: round(
                sum(1 for r in results if r.passed) / len(results), 3
            )
            for task_id, results in self.per_task().items()
        }
        return payload


def _collect_observation(
    service: Any, consultation_id: str, turn_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Assemble the facts the checks are allowed to look at."""
    consultation = service.get_consultation(consultation_id)
    events = service.repository.list_audit_events(consultation_id)
    versions = service.repository.list_itinerary_versions(consultation_id)

    latest_version = versions[-1] if versions else None
    payload = (latest_version or {}).get("payload") or {}
    budget = payload.get("budget") or {}

    event_types = [event.event_type for event in events]
    tool_events = [
        event for event in events if event.event_type in ("TOOL_EXECUTED", "TOOL_FAILED")
    ]
    tool_names = [
        (event.details or {}).get("tool_name")
        for event in tool_events
        if (event.details or {}).get("tool_name")
    ]

    return {
        "consultation_id": consultation_id,
        "trip_requirements": consultation.trip_requirements,
        "last_status": turn_results[-1].get("status") if turn_results else None,
        "statuses": [result.get("status") for result in turn_results],
        "version_count": len(versions),
        "has_version": latest_version is not None,
        "budget": {
            "total": budget.get("total"),
            "limit": budget.get("budget_limit"),
            "within_budget": budget.get("within_budget"),
            "nights": budget.get("nights"),
        },
        "event_types": event_types,
        "audit_event_count": len(events),
        "tool_call_count": len(tool_names),
        "tool_names": tool_names,
        "approval_count": len(
            [
                event
                for event in events
                if event.event_type == "AGENT_WAITING_FOR_APPROVAL"
            ]
        ),
        "booking_count": event_types.count("BOOKING_CREATED"),
        # 治理停止的原因：用于区分"模型做得不好"和"被预算/步数/超时拦下"。
        "tool_budget_exceeded": "TURN_STOPPED_BY_BUDGET" in event_types,
        "step_limit_exceeded": "TURN_STOPPED_BY_STEP_LIMIT" in event_types,
        "turn_timeout": "TURN_STOPPED_BY_TIMEOUT" in event_types,
    }


def _date_tokens(trip_date) -> dict[str, str]:
    """Placeholders a task may use so it never hard-codes a calendar date."""
    from datetime import timedelta

    end_date = trip_date + timedelta(days=2)
    return {
        "trip_date": f"{trip_date.year}年{trip_date.month}月{trip_date.day}日",
        "trip_date_end": f"{end_date.month}月{end_date.day}日",
        "trip_date_iso": trip_date.isoformat(),
        "trip_date_end_iso": end_date.isoformat(),
    }


def _reset_async_http_clients() -> None:
    """Drop cached httpx clients between tasks.

    Each task runs in its own ``asyncio.run``, which closes that loop. The
    OpenAI client caches its httpx transport with ``lru_cache`` keyed only by
    base_url/options, so the next task reuses a connection pool bound to the
    loop that just closed and fails with "Event loop is closed".

    Clearing the cache per task keeps the evaluation runnable until the
    persistent-service work (schedule 6.4) removes the per-request loop
    entirely. It is a workaround scoped to the runner, not a fix to the model
    layer.
    """
    try:
        from langchain_openai.chat_models import _client_utils
    except ImportError:  # pragma: no cover - 取决于依赖版本
        return
    # 只清缓存不够：被丢弃的 AsyncClient 会在 GC 时于已关闭的 loop 上调用
    # aclose()，刷出一片 "Event loop is closed"。必须先记下实例，再在
    # 当前（仍存活的）loop 里主动关掉它们。
    stale: list[Any] = []
    for name in ("_cached_async_httpx_client", "_cached_sync_httpx_client"):
        cached = getattr(_client_utils, name, None)
        if cached is None:
            continue
        if hasattr(cached, "cache_info") and hasattr(cached, "cache_clear"):
            cache = getattr(cached, "cache", None)
            if isinstance(cache, dict):
                stale.extend(cache.values())
            cached.cache_clear()
    if not stale:
        return
    import asyncio

    async def _close_all() -> None:
        for client in stale:
            aclose = getattr(client, "aclose", None)
            if aclose is None:
                continue
            try:
                await aclose()
            except Exception:  # noqa: BLE001 - 关闭失败不应影响评测
                pass

    try:
        asyncio.run(_close_all())
    except RuntimeError:
        # 已经在 loop 里（例如被评测代码直接调用）：交给 GC，不影响结果。
        pass


def _render(template: str, tokens: dict[str, str]) -> str:
    rendered = template
    for key, value in tokens.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _render_task(task: EvalTask, tokens: dict[str, str]) -> EvalTask:
    """Fill date placeholders so a task stays valid on any run day."""
    from dataclasses import replace

    rendered_checks = tuple(
        replace(check, value=_render(check.value, tokens))
        if isinstance(check.value, str)
        else check
        for check in task.checks
    )
    return replace(
        task,
        message=_render(task.message, tokens),
        followups=tuple(_render(item, tokens) for item in task.followups),
        checks=rendered_checks,
    )


def run_task(
    task: EvalTask,
    *,
    env_file: str | Path | None = None,
    model_name: str | None = None,
) -> TaskResult:
    """Run one task in an isolated database and evaluate its checks."""
    started = time.monotonic()
    _reset_async_http_clients()
    workdir = Path(tempfile.mkdtemp(prefix=f"eval-{task.id}-"))
    service = create_consultation_service(
        workdir / "eval.db", checkpoint_path=workdir / "eval_ck.sqlite"
    )
    # 数据必须按任务实际要用的日期 seed，否则样本库的固定日期一旦过期，
    # 所有任务都会因为"查不到交通"而失败，看起来像是模型退化。
    # seed 进 repository 的库：create_travel_tools 用的就是它。
    trip_date = prepare_eval_database(service.repository.database_path)
    # 任务里的日期是占位符，跑之前填成"本次实际 seed 的日期"，
    # 这样任务定义不会随日历过期。
    task = _render_task(task, _date_tokens(trip_date))
    consultation = service.create_consultation(
        "eval-consultant", "评测客户", task.message
    )

    turn_results: list[dict[str, Any]] = []
    error: str | None = None
    try:
        turn_results.append(
            service.chat_consultation(
                consultation.id, task.message, env_file=env_file, model_name=model_name
            )
        )
        for followup in task.followups:
            # 每一轮都是一次 asyncio.run，loop 会关闭，因此每轮之间同样要
            # 清掉绑定在旧 loop 上的 httpx 连接池。
            _reset_async_http_clients()
            turn_results.append(
                service.chat_consultation(
                    consultation.id, followup, env_file=env_file, model_name=model_name
                )
            )
    except Exception as exception:  # noqa: BLE001 - 评测失败必须被记录，不能中断整批
        error = f"{type(exception).__name__}: {exception}"

    observation = _collect_observation(service, consultation.id, turn_results)
    observation["eval_trip_date"] = trip_date.isoformat()
    outcomes = apply_checks(task.checks, observation)
    failures = hard_failures(outcomes)
    result = TaskResult(
        task_id=task.id,
        passed=not failures and error is None,
        observation=observation,
        outcomes=outcomes,
        error=error,
        duration_seconds=time.monotonic() - started,
    )
    # 失败任务的证据必须留下来：报告的 observation 只有汇总事实，
    # 真要排障还得看那一轮的审计记录和完整工具调用。通过的任务则删掉，
    # 避免一次跑批在 /tmp 里堆几十个 sqlite 文件。
    if result.passed:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        result.observation["workdir"] = str(workdir)
    return result


def run_tasks(
    tasks: list[EvalTask],
    *,
    env_file: str | Path | None = None,
    model_name: str | None = None,
    progress: Callable[[TaskResult], None] | None = None,
    repeat: int = 1,
    on_result: Callable[[EvalReport], None] | None = None,
) -> EvalReport:
    """Run every task sequentially and collect a report.

    Sequential on purpose: the point is a reproducible per-task result, not
    throughput. A bundled runner would also fight the provider rate limiter.

    ``repeat`` exists because a single pass cannot distinguish "the model does
    this reliably" from "the model did this once". Real models are stochastic,
    so a baseline is only meaningful with several samples per task.

    ``on_result`` is called after every task with the partial report. A run
    costs minutes of real model calls, so an interrupted run must not discard
    what it already measured - pressing Ctrl+C on task 40 of 45 should still
    leave a usable report behind.
    """
    from datetime import datetime

    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    resolved_model = model_name
    if not resolved_model:
        # 基线必须能回答"这是哪个模型跑出来的"，否则两次报告无法比较。
        # CLI 未显式传 --model-name 时，从同一份配置里解析实际模型名。
        try:
            from ..agent import load_agent_settings

            resolved_model = load_agent_settings(env_file).model_name
        except Exception:  # noqa: BLE001 - 配置缺失不应让评测无法进行
            resolved_model = "unknown"
    report = EvalReport(
        model=resolved_model or "unknown",
        started_at=datetime.now().isoformat(timespec="seconds"),
        repeat=repeat,
    )
    for _ in range(repeat):
        for task in tasks:
            result = run_task(task, env_file=env_file, model_name=model_name)
            report.results.append(result)
            # 每完成一个任务就报告。此前这个回调在内层循环之外，
            # repeat>1 时只打印每轮的最后一条，前面的结果看不到。
            if progress is not None:
                progress(result)
            if on_result is not None:
                on_result(report)
    return report


def write_report(report: EvalReport, directory: str | Path) -> Path:
    """Persist a report so two runs can be compared later."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    stamp = report.started_at.replace(":", "").replace("-", "")
    target = path / f"{stamp}.json"
    target.write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2), "utf-8"
    )
    return target
