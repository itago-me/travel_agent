"""评测体系：任务定义、断言与执行。

对应 schedule.md 阶段 7.1。设计原则见各子模块的模块文档，核心是
**硬断言与软指标分离**，以及**断言只看结构化事实，不看模型的自然语言**。
"""

from .checks import CheckOutcome, apply_check, apply_checks, hard_failures, soft_score
from .runner import EvalReport, TaskResult, run_task, run_tasks, write_report
from .tasks import HARD, SOFT, Check, EvalTask, load_tasks, parse_task

__all__ = [
    "HARD",
    "SOFT",
    "Check",
    "CheckOutcome",
    "EvalReport",
    "EvalTask",
    "TaskResult",
    "apply_check",
    "apply_checks",
    "hard_failures",
    "load_tasks",
    "parse_task",
    "run_task",
    "run_tasks",
    "soft_score",
    "write_report",
]
