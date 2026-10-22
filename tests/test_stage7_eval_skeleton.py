"""阶段 7.1：评测骨架。

骨架本身必须被测——否则"评测不准"这件事没人发现。这里验证三件事：

1. 任务定义与加载（含占位符、重复 id、非法字段）；
2. 断言引擎的判定语义（硬/软、各 op、缺失路径）；
3. 执行器把事实正确装进 observation，并让硬断言决定成败。

不验证模型水平——那是 7.2 用真实模型跑基线时才做的事。
"""

import json

import pytest

from travel_agent.evals import (
    HARD,
    SOFT,
    Check,
    EvalTask,
    apply_check,
    apply_checks,
    hard_failures,
    load_tasks,
    parse_task,
    soft_score,
)
from travel_agent.evals.checks import CheckOutcome, resolve_path
from travel_agent.evals.fixtures import (
    eval_trip_date,
    prepare_eval_database,
    prompt_for_date,
)
from travel_agent.evals.runner import _date_tokens, _render, _render_task


# --- 任务加载 ---------------------------------------------------------------


def test_load_bundled_task_set():
    tasks = load_tasks("evals/tasks")
    assert len(tasks) >= 10, "任务集应至少覆盖 10 条，才有最基本的代表性"
    ids = [task.id for task in tasks]
    assert len(ids) == len(set(ids))


def test_every_task_has_at_least_one_hard_check():
    """没有硬断言的任务无法判定成败，只会永远"通过"。"""
    for task in load_tasks("evals/tasks"):
        assert task.hard_checks, f"task {task.id} has no hard check"


def test_check_kind_split():
    task = parse_task(
        {
            "id": "t",
            "message": "m",
            "checks": [
                {"kind": "hard", "name": "a", "op": "present", "path": "x"},
                {"kind": "soft", "name": "b", "op": "present", "path": "y"},
            ],
        }
    )
    assert len(task.hard_checks) == 1
    assert len(task.soft_checks) == 1


def test_invalid_kind_is_rejected():
    with pytest.raises(ValueError, match="check kind"):
        Check(kind="maybe", name="x", op="present")


def test_missing_check_field_names_the_task():
    with pytest.raises(ValueError, match="missing field"):
        parse_task({"id": "t", "message": "m", "checks": [{"kind": "hard"}]})


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "dup.json"
    path.write_text(
        json.dumps(
            [
                {"id": "same", "message": "a"},
                {"id": "same", "message": "b"},
            ]
        ),
        "utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_tasks(tmp_path)


def test_dict_wrapped_task_file_is_accepted(tmp_path):
    (tmp_path / "w.json").write_text(
        json.dumps({"tasks": [{"id": "x", "message": "m"}]}), "utf-8"
    )
    assert [task.id for task in load_tasks(tmp_path)] == ["x"]


# --- 占位符 ---------------------------------------------------------------


def test_date_tokens_render_a_natural_prompt():
    from datetime import date

    tokens = _date_tokens(date(2026, 11, 3))
    assert tokens["trip_date"] == "2026年11月3日"
    assert tokens["trip_date_iso"] == "2026-11-03"
    assert tokens["trip_date_end_iso"] == "2026-11-05"


def test_task_rendering_fills_message_and_checks():
    from datetime import date

    task = EvalTask(
        id="t",
        message="{trip_date_iso}出发",
        checks=(
            Check(kind=HARD, name="d", op="equals", path="x", value="{trip_date_iso}"),
        ),
    )
    rendered = _render_task(task, _date_tokens(date(2026, 11, 3)))

    assert rendered.message == "2026-11-03出发"
    assert rendered.checks[0].value == "2026-11-03"


def test_render_leaves_unknown_tokens_alone():
    assert _render("keep {unknown}", {}) == "keep {unknown}"


def test_bundled_tasks_have_no_hardcoded_dates():
    """任务不能写死日历日期——那会在某天之后静默失效。

    真正要防的是"写死一个具体年份/日期"，而不是"每条都必须带日期"：
    相对日期任务（"明天"）和缺日期任务（"玩几天"）本来就刻意不带日期。
    """
    import re

    hardcoded = re.compile(r"20\d{2}年|20\d{2}-\d{2}-\d{2}")
    for task in load_tasks("evals/tasks"):
        assert not hardcoded.search(task.message), (
            f"task {task.id} hard-codes a date in its message"
        )
        for check in task.checks:
            if isinstance(check.value, str):
                assert not hardcoded.search(check.value), (
                    f"task {task.id} check {check.name} hard-codes a date"
                )


# --- fixtures -------------------------------------------------------------


def test_eval_trip_date_is_in_the_future():
    from datetime import date

    assert eval_trip_date() > date.today()


def test_prepare_eval_database_seeds_transport_for_that_date(tmp_path):
    import sqlite3

    database = tmp_path / "t.db"
    seeded = prepare_eval_database(database)
    with sqlite3.connect(database) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM travel_transport_options WHERE departure_date = ?",
            (seeded.isoformat(),),
        ).fetchone()[0]

    assert count > 0, "评测日期必须有交通数据，否则版本永远生成不出来"


def test_prompt_for_date_matches_consultant_style():
    from datetime import date

    assert prompt_for_date(date(2026, 11, 3)) == "2026年11月3日"


# --- 断言引擎 -------------------------------------------------------------


def test_resolve_path_reads_nested_dicts():
    observation = {"a": {"b": {"c": 1}}, "list": [10, 20]}
    assert resolve_path(observation, "a.b.c") == 1
    assert resolve_path(observation, "list.1") == 20
    assert resolve_path(observation, "") is observation


def test_resolve_path_returns_none_for_missing():
    assert resolve_path({"a": 1}, "a.b.c") is None
    assert resolve_path({"a": 1}, "nope") is None


def test_ops():
    observation = {"n": 5, "s": "abc", "none": None, "yes": True, "no": False}
    assert apply_check(Check(HARD, "x", "present", "n"), observation).passed
    assert apply_check(Check(HARD, "x", "absent", "none"), observation).passed
    assert apply_check(Check(HARD, "x", "truthy", "yes"), observation).passed
    assert apply_check(Check(HARD, "x", "falsy", "no"), observation).passed
    assert apply_check(Check(HARD, "x", "equals", "n", 5), observation).passed
    assert apply_check(Check(HARD, "x", "contains", "s", "b"), observation).passed
    assert apply_check(Check(HARD, "x", "gte", "n", 5), observation).passed
    assert apply_check(Check(HARD, "x", "lte", "n", 5), observation).passed
    assert apply_check(Check(HARD, "x", "within", "n", [1, 10]), observation).passed


def test_failing_op_reports_actual_and_expected():
    outcome = apply_check(
        Check(HARD, "n", "equals", "n", 9), {"n": 5}
    )
    assert not outcome.passed
    assert outcome.actual == 5
    assert outcome.expected == 9


def test_gte_on_non_numeric_fails_instead_of_raising():
    outcome = apply_check(Check(HARD, "x", "gte", "s", 1), {"s": "abc"})
    assert not outcome.passed
    assert "not comparable" in outcome.detail


def test_unknown_op_is_reported():
    outcome = apply_check(Check(HARD, "x", "wat", "n"), {"n": 1})
    assert not outcome.passed
    assert "unknown check op" in outcome.detail


def test_hard_failures_ignores_soft():
    outcomes = [
        CheckOutcome(kind=HARD, name="a", passed=True),
        CheckOutcome(kind=SOFT, name="b", passed=False),
    ]
    assert hard_failures(outcomes) == []


def test_hard_failure_is_reported():
    outcomes = [CheckOutcome(kind=HARD, name="a", passed=False)]
    assert len(hard_failures(outcomes)) == 1


def test_soft_score_is_fraction_of_passing_soft_checks():
    outcomes = [
        CheckOutcome(kind=SOFT, name="a", passed=True),
        CheckOutcome(kind=SOFT, name="b", passed=False),
        CheckOutcome(kind=HARD, name="c", passed=False),
    ]
    assert soft_score(outcomes) == 0.5, "硬断言不参与软分计算"


def test_soft_score_is_none_without_soft_checks():
    assert soft_score([CheckOutcome(kind=HARD, name="a", passed=True)]) is None


def test_apply_checks_runs_all():
    checks = (
        Check(HARD, "a", "present", "x"),
        Check(SOFT, "b", "present", "y"),
    )
    assert len(apply_checks(checks, {"x": 1})) == 2


# --- 报告 -----------------------------------------------------------------


def test_report_counts_and_serialisation():
    from travel_agent.evals.runner import EvalReport, TaskResult

    report = EvalReport(
        results=[
            TaskResult(task_id="a", passed=True, observation={}),
            TaskResult(task_id="b", passed=False, observation={}),
        ],
        model="m",
        started_at="2026-11-03T10:00:00",
    )
    payload = report.as_dict()

    assert report.passed == 1
    assert report.failed == 1
    summary = payload["summary"]
    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 1


def test_report_is_written_to_disk(tmp_path):
    from travel_agent.evals.runner import EvalReport, TaskResult, write_report

    report = EvalReport(
        results=[TaskResult(task_id="a", passed=True, observation={"x": 1})],
        model="m",
        started_at="2026-11-03T10:00:00",
    )
    path = write_report(report, tmp_path)

    assert path.exists()
    assert json.loads(path.read_text("utf-8"))["tasks"][0]["task_id"] == "a"


# --- 稳定性（重复运行）------------------------------------------------------


def _report_with(outcomes: dict[str, list[bool]]) -> "EvalReport":
    from travel_agent.evals.runner import EvalReport, TaskResult

    report = EvalReport(repeat=3)
    for task_id, passes in outcomes.items():
        for passed in passes:
            report.results.append(TaskResult(task_id=task_id, passed=passed, observation={}))
    return report


def test_stability_separates_flaky_from_consistent():
    """flaky 比 always_fail 更危险：后者是已知 bug，前者会悄悄侵蚀信任。"""
    report = _report_with(
        {
            "stable_pass": [True, True, True],
            "stable_fail": [False, False, False],
            "flaky": [True, False, True],
        }
    )
    stability = report.stability()

    assert stability["always_pass"] == ["stable_pass"]
    assert stability["always_fail"] == ["stable_fail"]
    assert stability["flaky"] == ["flaky"]


def test_task_pass_rates_are_reported():
    report = _report_with({"a": [True, True, False]})
    rates = report.as_dict()["summary"]["task_pass_rates"]

    assert rates["a"] == 0.667


def test_repeat_must_be_positive():
    from travel_agent.evals import run_tasks

    with pytest.raises(ValueError, match="repeat"):
        run_tasks([], repeat=0)


def test_single_repeat_has_no_flaky_bucket():
    report = _report_with({"a": [True], "b": [False]})
    stability = report.stability()

    assert stability["flaky"] == []
    assert stability["always_pass"] == ["a"]
    assert stability["always_fail"] == ["b"]


# --- 基线整理 -------------------------------------------------------------


def _sample_report() -> dict:
    def task(task_id, passed, tools, soft=None, seconds=5.0):
        entry = {
            "task_id": task_id,
            "passed": passed,
            "duration_seconds": seconds,
            "observation": {"tool_call_count": tools},
        }
        if soft is not None:
            entry["soft_score"] = soft
        return entry

    return {
        "model": "m",
        "stability": {"repeat": 2},
        "tasks": [
            task("solid", True, 4, 1.0),
            task("solid", True, 6, 1.0),
            task("shaky", True, 4, 0.5),
            task("shaky", False, 8, 1.0),
            task("broken", False, 2),
            task("broken", False, 3),
        ],
    }


def test_baseline_computes_pass_rates():
    from travel_agent.evals.baseline import build_baseline

    baseline = build_baseline(_sample_report())
    rows = {row["task_id"]: row for row in baseline["tasks"]}

    assert rows["solid"]["pass_rate"] == 1.0
    assert rows["shaky"]["pass_rate"] == 0.5
    assert rows["broken"]["pass_rate"] == 0.0


def test_baseline_flags_flaky_tasks_only():
    from travel_agent.evals.baseline import build_baseline

    rows = {row["task_id"]: row for row in build_baseline(_sample_report())["tasks"]}

    assert rows["shaky"]["flaky"] is True
    assert rows["solid"]["flaky"] is False
    assert rows["broken"]["flaky"] is False


def test_baseline_reports_tool_cost():
    from travel_agent.evals.baseline import build_baseline

    rows = {row["task_id"]: row for row in build_baseline(_sample_report())["tasks"]}

    assert rows["solid"]["avg_tool_calls"] == 5.0
    assert rows["solid"]["max_tool_calls"] == 6


def test_baseline_sorts_worst_first():
    """最差的排在最前面——基线是用来找问题的，不是用来欣赏的。"""
    from travel_agent.evals.baseline import build_baseline

    order = [row["task_id"] for row in build_baseline(_sample_report())["tasks"]]

    assert order[0] == "broken"


def test_format_baseline_marks_flaky_rows():
    from travel_agent.evals.baseline import build_baseline, format_baseline

    text = format_baseline(build_baseline(_sample_report()))

    assert "shaky" in text
    assert "!" in text
    assert "always_fail (1): broken" in text


# --- 报告对比与可追溯（7.4）------------------------------------------------


def _report(tasks: dict[str, tuple[bool, int]]) -> dict:
    return {
        "model": "m",
        "stability": {"repeat": 1},
        "tasks": [
            {
                "task_id": task_id,
                "passed": passed,
                "duration_seconds": 5.0,
                "observation": {"tool_call_count": tools},
            }
            for task_id, (passed, tools) in tasks.items()
        ],
    }


def test_comparison_detects_regression():
    """对比要按任务看：总体通过数不变时，任务仍可能互相换位置。"""
    from travel_agent.evals.baseline import compare_reports

    comparison = compare_reports(
        _report({"a": (True, 4), "b": (False, 4)}),
        _report({"a": (False, 4), "b": (True, 4)}),
    )

    assert comparison["regressed"] == ["a"]
    assert comparison["fixed"] == ["b"]


def test_comparison_reports_tool_call_changes():
    from travel_agent.evals.baseline import compare_reports

    comparison = compare_reports(
        _report({"a": (True, 9)}), _report({"a": (True, 4)})
    )

    assert comparison["tool_changes"][0]["task_id"] == "a"
    assert comparison["tool_changes"][0]["delta"] == -5


def test_comparison_tracks_added_and_removed_tasks():
    from travel_agent.evals.baseline import compare_reports

    comparison = compare_reports(
        _report({"old": (True, 3)}), _report({"new": (True, 3)})
    )

    assert comparison["added"] == ["new"]
    assert comparison["removed"] == ["old"]


def test_comparison_without_differences_says_so():
    from travel_agent.evals.baseline import compare_reports, format_comparison

    comparison = compare_reports(_report({"a": (True, 4)}), _report({"a": (True, 4)}))

    assert format_comparison(comparison) == "no differences"


def test_latest_report_picks_newest(tmp_path):
    from travel_agent.cli import _latest_report

    (tmp_path / "20260101T000000.json").write_text("{}", "utf-8")
    (tmp_path / "20260202T000000.json").write_text("{}", "utf-8")

    assert _latest_report(tmp_path).name == "20260202T000000.json"


def test_latest_report_handles_empty_dir(tmp_path):
    from travel_agent.cli import _latest_report

    assert _latest_report(tmp_path / "missing") is None


def test_shared_only_suppresses_phantom_removals():
    """用 --only 跑一部分任务时，其余任务不是"被删除"，不该报噪音。"""
    from travel_agent.evals.baseline import compare_reports

    comparison = compare_reports(
        _report({"a": (True, 4), "b": (True, 4), "c": (True, 4)}),
        _report({"a": (True, 4)}),
        shared_only=True,
    )

    assert comparison["removed"] == []


def test_shared_only_still_detects_real_regression():
    from travel_agent.evals.baseline import compare_reports

    comparison = compare_reports(
        _report({"a": (True, 4), "b": (True, 4)}),
        _report({"a": (False, 4)}),
        shared_only=True,
    )

    assert comparison["regressed"] == ["a"]


# --- 中断与进度（7.4 后续）--------------------------------------------------


def test_progress_fires_for_every_task_across_repeats():
    """progress 在内层循环之外时，repeat>1 只打印每轮最后一条。"""
    from travel_agent.evals import run_tasks

    from travel_agent.evals.runner import TaskResult

    seen: list[str] = []
    tasks = [EvalTask(id=f"t{i}", message="m") for i in range(3)]

    import travel_agent.evals.runner as runner_module

    original = runner_module.run_task
    try:
        runner_module.run_task = lambda task, **kw: TaskResult(
            task_id=task.id, passed=True, observation={}
        )
        run_tasks(tasks, repeat=2, progress=lambda r: seen.append(r.task_id))
    finally:
        runner_module.run_task = original

    assert len(seen) == 6, "3 个任务 x 2 轮 = 6 次回调"


def test_on_result_receives_partial_report_after_each_task():
    """增量落盘是中断时保住结果的前提。"""
    from travel_agent.evals import run_tasks

    from travel_agent.evals.runner import TaskResult

    sizes: list[int] = []
    tasks = [EvalTask(id=f"t{i}", message="m") for i in range(3)]

    import travel_agent.evals.runner as runner_module

    original = runner_module.run_task
    try:
        runner_module.run_task = lambda task, **kw: TaskResult(
            task_id=task.id, passed=True, observation={}
        )
        run_tasks(tasks, on_result=lambda rep: sizes.append(len(rep.results)))
    finally:
        runner_module.run_task = original

    assert sizes == [1, 2, 3]


def test_partial_report_is_usable():
    """中断后拿到的部分报告必须能正常序列化和统计。"""
    from travel_agent.evals.runner import EvalReport, TaskResult

    partial = EvalReport(repeat=1)
    partial.results.append(TaskResult(task_id="a", passed=True, observation={}))

    payload = partial.as_dict()

    assert payload["summary"]["total"] == 1
    assert payload["stability"]["always_pass"] == ["a"]


# --- 环境预检（防误导报告）--------------------------------------------------


def test_environment_check_passes_without_proxy(monkeypatch):
    """没有代理时不该报任何问题。"""
    import importlib.util

    from travel_agent.evals import fixtures

    for var in ("all_proxy", "ALL_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.delenv(var, raising=False)

    assert fixtures.check_eval_environment() == []


def test_environment_check_rejects_socks_without_socksio(monkeypatch):
    """SOCKS 代理缺 socksio 时，每个任务都会 ImportError——必须提前拦下。"""
    import importlib.util
    import types

    from travel_agent.evals import fixtures

    for var in ("ALL_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("all_proxy", "socks5://127.0.0.1:7897")

    # 函数内部 import importlib.util 拿到的是标准库模块本体，
    # 所以补丁要打在 importlib.util 上，而不是 fixtures 模块里。
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name):
        if name == "socksio":
            return None
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    problems = fixtures.check_eval_environment()

    assert len(problems) == 1
    assert "socksio" in problems[0]
    assert "all_proxy" in problems[0]


def test_environment_check_ignores_http_proxy(monkeypatch):
    """http 代理不需要 socksio，不该误报。

    注意要先清掉 all_proxy：shell 里若同时有 socks 代理会泄漏进来，
    这本身就是测试环境最常见的坑。
    """
    from travel_agent.evals import fixtures

    for var in ("all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:7897")

    assert fixtures.check_eval_environment() == []
