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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
