import argparse
import json
from pathlib import Path

from .application import create_consultation_service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="travel-agent")
    parser.add_argument("--database", default="data/travel_agent.db")
    parser.add_argument("--checkpoint-database")

    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--consultant-id", required=True)
    start.add_argument("--customer-id")
    start.add_argument("--customer-name", required=True)
    start.add_argument("--message", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("consultation_id")

    subparsers.add_parser("sensitive-tools")

    eval_parser = subparsers.add_parser(
        "eval", help="run the evaluation task suite against a real model"
    )
    eval_parser.add_argument("--tasks", default="evals/tasks")
    eval_parser.add_argument("--results", default="evals/results")
    eval_parser.add_argument("--env-file")
    eval_parser.add_argument("--model-name")
    eval_parser.add_argument(
        "--only", action="append", default=[], help="run only these task ids"
    )
    eval_parser.add_argument(
        "--json", action="store_true", help="print the full report instead of a summary"
    )
    eval_parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run the suite N times to measure stability, not just one sample",
    )

    pending = subparsers.add_parser("pending")
    pending.add_argument("consultation_id")

    approve = subparsers.add_parser("approve")
    approve.add_argument("approval_id")
    approve.add_argument("--decided-by", required=True)
    approve.add_argument("--reason")
    approve.add_argument(
        "--edit-json",
        help="编辑后的工具参数 JSON；提供时将批准并附带修改后的参数",
    )
    approve.add_argument(
        "--no-resume",
        action="store_true",
        help="只记录决定，不继续执行被暂停的线程",
    )

    reject = subparsers.add_parser("reject")
    reject.add_argument("approval_id")
    reject.add_argument("--decided-by", required=True)
    reject.add_argument("--reason", required=True)
    reject.add_argument("--no-resume", action="store_true")

    resume_approval = subparsers.add_parser("resume-approval")
    resume_approval.add_argument("consultation_id")

    chat = subparsers.add_parser("chat")
    chat.add_argument("consultation_id")
    chat.add_argument("--message", required=True)
    chat.add_argument("--env-file", default=".env")
    chat.add_argument("--model-name")
    chat.add_argument("--api-key")
    chat.add_argument("--base-url")

    itineraries = subparsers.add_parser("itineraries")
    itineraries.add_argument("consultation_id")

    select = subparsers.add_parser("select-itinerary")
    select.add_argument("consultation_id")
    select.add_argument("version_id")

    preferences = subparsers.add_parser("preferences")
    preferences.add_argument("customer_id")

    set_preference = subparsers.add_parser("set-preference")
    set_preference.add_argument("customer_id")
    set_preference.add_argument("preference_key")
    set_preference.add_argument("--value-json", required=True)
    set_preference.add_argument("--source", default="explicit")

    delete_preference = subparsers.add_parser("delete-preference")
    delete_preference.add_argument("customer_id")
    delete_preference.add_argument("preference_key")
    return parser


def _fail(parser: argparse.ArgumentParser, message: str) -> int:
    """Report a business-level failure as a clean CLI error, not a traceback."""
    parser.error(message)
    return 2


def _latest_report(results_dir: str | Path):
    """Return the most recent report in ``results_dir``, or None."""
    directory = Path(results_dir)
    if not directory.is_dir():
        return None
    candidates = sorted(directory.glob("*.json"))
    return candidates[-1] if candidates else None


def _run_eval(args: argparse.Namespace) -> int:
    """Run the evaluation suite and print a summary.

    Exit code carries the verdict: 0 when every task passed, 1 otherwise, so the
    command can gate later automation without parsing text.
    """
    from .evals import load_tasks, run_tasks, write_report
    from .evals.fixtures import check_eval_environment

    # 先确认环境能跑。跑不了的时候产出"15 条全失败"的报告，比不跑更糟：
    # 它看起来和模型集体退化一模一样，会让人往完全错误的方向排查。
    problems = check_eval_environment()
    if problems:
        print("cannot run evaluation:")
        for problem in problems:
            print(f"  - {problem}")
        return 2

    tasks = load_tasks(args.tasks)
    if args.only:
        wanted = set(args.only)
        tasks = [task for task in tasks if task.id in wanted]
        if not tasks:
            print(f"no tasks matched --only {sorted(wanted)}")
            return 2

    def announce(result) -> None:
        mark = "PASS" if result.passed else "FAIL"
        score = result.soft_score
        suffix = f" soft={score:.2f}" if score is not None else ""
        print(f"[{mark}] {result.task_id} ({result.duration_seconds:.1f}s){suffix}")
        for outcome in result.outcomes:
            if outcome.passed or outcome.kind != "hard":
                continue
            print(
                f"       hard failed: {outcome.name} "
                f"(actual={outcome.actual!r} expected={outcome.expected!r})"
            )
        if result.error:
            print(f"       error: {result.error}")

    # 写报告前先记下"上一次"是谁：评测的价值一半来自前后对比，
    # 每次都要人手动指定旧报告，就没人会真的去比。
    previous_path = _latest_report(args.results)

    # 一次评测是几分钟的真实模型调用。中途 Ctrl+C 不该让已完成的部分白跑，
    # 所以每完成一个任务就落一次盘；同一份报告覆盖写，跑完仍是单份结果。
    latest: dict[str, object] = {}

    def persist(current) -> None:
        latest["report"] = current
        write_report(current, args.results)

    try:
        report = run_tasks(
            tasks,
            env_file=args.env_file,
            model_name=args.model_name,
            progress=announce,
            repeat=args.repeat,
            on_result=persist,
        )
    except KeyboardInterrupt:
        # 中断不是异常状态：已经跑完的任务照样有效，照常汇报。
        from .evals.runner import EvalReport

        print("\ninterrupted - reporting completed tasks only")
        # 被打断的那一轮还挂着 httpx 连接；在进程退出前关掉，
        # 否则 GC 会在已关闭的 loop 上 aclose() 并刷一片报错。
        try:
            from .evals.runner import _reset_async_http_clients

            _reset_async_http_clients()
        except Exception:  # noqa: BLE001 - 清理失败不该盖住中断本身
            pass
        report = latest.get("report") or EvalReport()
        if not report.results:
            print("no task finished before the interrupt; nothing to report")
            return 130
    path = write_report(report, args.results)

    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print()
        print(
            f"total={len(report.results)} passed={report.passed} failed={report.failed}"
        )
        average = report.soft_average
        if average is not None:
            print(f"soft average={average:.2f}")
        if previous_path is not None:
            from .evals.baseline import compare_reports, format_comparison

            try:
                previous = json.loads(previous_path.read_text("utf-8"))
                # 只跑了一部分任务时（--only），"其余任务消失"是无意义的噪音，
                # 甚至会被误读成任务被删。此时只比较两次都跑过的任务。
                comparison = compare_reports(previous, report.as_dict(), shared_only=True)
                if args.only:
                    print(f"\nvs {previous_path.name} (only tasks run here):")
                else:
                    print(f"\nvs {previous_path.name}:")
                print(format_comparison(comparison))
            except (OSError, ValueError) as error:
                print(f"\n(skipped comparison with {previous_path.name}: {error})")
        if report.repeat > 1:
            stability = report.stability()
            print(
                f"stability: always_pass={len(stability['always_pass'])} "
                f"always_fail={len(stability['always_fail'])} "
                f"flaky={len(stability['flaky'])}"
            )
            for task_id in stability["flaky"]:
                rates = report.as_dict()["summary"]["task_pass_rates"]
                print(f"  flaky: {task_id} pass_rate={rates[task_id]}")
            for task_id in stability["always_fail"]:
                print(f"  always_fail: {task_id}")
        print(f"report={path}")
    return 0 if report.failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "eval":
        return _run_eval(args)

    service = create_consultation_service(
        Path(args.database),
        Path(args.checkpoint_database) if args.checkpoint_database else None,
    )
    if args.command == "sensitive-tools":
        from .langchain_tools import create_sensitive_booking_tools

        tools = create_sensitive_booking_tools(Path(args.database))
        print(
            json.dumps(
                [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "permission": tool.metadata["permission"],
                        "requires_human_approval": tool.metadata[
                            "requires_human_approval"
                        ],
                        "input_schema": tool.args_schema.model_json_schema(),
                    }
                    for tool in tools
                ],
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "start":
        consultation = service.create_consultation(
            args.consultant_id,
            args.customer_name,
            args.message,
            customer_id=args.customer_id,
        )
        print(
            json.dumps(
                {
                    "consultation_id": consultation.id,
                    "thread_id": consultation.thread_id,
                    "customer_id": consultation.customer_id,
                }
            )
        )
        return 0
    if args.command == "pending":
        try:
            approvals = service.list_pending_approvals(args.consultation_id)
        except KeyError:
            return _fail(parser, f"咨询不存在: {args.consultation_id}")
        print(
            json.dumps(
                {
                    "consultation_id": args.consultation_id,
                    "status": "WAITING_FOR_APPROVAL"
                    if approvals
                    else "NO_PENDING_APPROVAL",
                    "approval_requests": [approval.as_dict() for approval in approvals],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "approve":
        edited_arguments = None
        if args.edit_json is not None:
            try:
                edited_arguments = json.loads(args.edit_json)
            except json.JSONDecodeError:
                parser.error("--edit-json 必须是有效 JSON 对象")
            if not isinstance(edited_arguments, dict):
                parser.error("--edit-json 必须是 JSON 对象")
        decision = "EDIT" if edited_arguments is not None else "APPROVE"
        try:
            result = service.decide_approval(
                args.approval_id,
                decision,
                decided_by=args.decided_by,
                reason=args.reason,
                edited_arguments=edited_arguments,
                resume=not args.no_resume,
            )
        except KeyError as error:
            return _fail(parser, f"审批请求不存在: {args.approval_id}")
        except ValueError as error:
            return _fail(parser, str(error))
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "reject":
        try:
            result = service.reject_approval(
                args.approval_id,
                decided_by=args.decided_by,
                reason=args.reason,
                resume=not args.no_resume,
            )
        except KeyError:
            return _fail(parser, f"审批请求不存在: {args.approval_id}")
        except ValueError as error:
            return _fail(parser, str(error))
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "resume-approval":
        try:
            result = service.resume_approved_consultation(args.consultation_id)
        except KeyError:
            return _fail(parser, f"咨询不存在: {args.consultation_id}")
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "preferences":
        preferences = service.list_customer_preferences(args.customer_id)
        print(
            json.dumps(
                [preference.as_dict() for preference in preferences],
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "set-preference":
        try:
            value = json.loads(args.value_json)
        except json.JSONDecodeError:
            parser.error("--value-json 必须是有效 JSON")
        preference = service.set_customer_preference(
            args.customer_id,
            args.preference_key,
            value,
            source=args.source,
        )
        print(json.dumps(preference.as_dict(), ensure_ascii=False))
        return 0
    if args.command == "delete-preference":
        deleted = service.delete_customer_preference(
            args.customer_id, args.preference_key
        )
        print(
            json.dumps(
                {
                    "customer_id": args.customer_id,
                    "preference_key": args.preference_key,
                    "deleted": deleted,
                },
                ensure_ascii=False,
            )
        )
        return 0
    consultation = service.get_consultation(args.consultation_id)
    if args.command == "itineraries":
        print(
            json.dumps(
                service.repository.list_itinerary_versions(args.consultation_id),
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "select-itinerary":
        print(
            json.dumps(
                service.select_itinerary(args.consultation_id, args.version_id),
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "chat":
        result = service.chat_consultation(
            args.consultation_id,
            args.message,
            model_name=args.model_name,
            api_key=args.api_key,
            base_url=args.base_url,
            env_file=args.env_file,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    print(
        json.dumps(
            {
                "consultation_id": consultation.id,
                "thread_id": consultation.thread_id,
                "customer_id": consultation.customer_id,
                "customer_name": consultation.customer_name,
                "status": consultation.status,
                "approval_requests": [
                    approval.as_dict()
                    for approval in service.repository.list_pending_approval_requests(
                        consultation.id
                    )
                ],
                "messages": [message.as_dict() for message in consultation.messages],
                "trip_requirements": consultation.trip_requirements,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
