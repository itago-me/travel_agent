import argparse
import json
from pathlib import Path

from .application import create_consultation_service
from .tool_gateway import create_planning_tool_gateway


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="travel-agent")
    parser.add_argument("--database", default="data/travel_agent.db")
    parser.add_argument("--checkpoint-database")

    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--consultant-id", required=True)
    start.add_argument("--customer-name", required=True)
    start.add_argument("--message", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("consultation_id")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("consultation_id")

    resume = subparsers.add_parser("resume")
    resume.add_argument("consultation_id")
    resume.add_argument("--message", required=True)

    subparsers.add_parser("tools")

    search_options = subparsers.add_parser("search-options")
    search_options.add_argument("consultation_id")

    calculate_options = subparsers.add_parser("calculate-options")
    calculate_options.add_argument("consultation_id")

    plan = subparsers.add_parser("plan")
    plan.add_argument("consultation_id")
    plan.add_argument("--max-tool-calls", type=int, default=6)

    revise = subparsers.add_parser("revise")
    revise.add_argument("consultation_id")
    revise.add_argument("--message", required=True)

    itineraries = subparsers.add_parser("itineraries")
    itineraries.add_argument("consultation_id")

    compare = subparsers.add_parser("compare-itineraries")
    compare.add_argument("consultation_id")

    select = subparsers.add_parser("select-itinerary")
    select.add_argument("consultation_id")
    select.add_argument("version_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = create_consultation_service(
        Path(args.database),
        Path(args.checkpoint_database) if args.checkpoint_database else None,
    )
    if args.command == "tools":
        tools = create_planning_tool_gateway().list_tools()
        print(
            json.dumps(
                [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "permission": tool.permission.value,
                        "input_schema": tool.input_model.model_json_schema(),
                    }
                    for tool in tools
                ],
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "start":
        consultation = service.create_consultation(
            args.consultant_id, args.customer_name, args.message
        )
        print(
            json.dumps(
                {
                    "consultation_id": consultation.id,
                    "thread_id": consultation.thread_id,
                }
            )
        )
        return 0
    consultation = service.get_consultation(args.consultation_id)
    if args.command == "itineraries":
        print(json.dumps(service.repository.list_itinerary_versions(args.consultation_id), ensure_ascii=False))
        return 0
    if args.command == "compare-itineraries":
        print(json.dumps(service.compare_itineraries(args.consultation_id), ensure_ascii=False))
        return 0
    if args.command == "select-itinerary":
        print(json.dumps(service.select_itinerary(args.consultation_id, args.version_id), ensure_ascii=False))
        return 0
    if args.command == "plan":
        result = service.plan_consultation(
            args.consultation_id,
            max_tool_calls=args.max_tool_calls,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "revise":
        result = service.revise_consultation(args.consultation_id, args.message)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "search-options":
        result = service.search_options(args.consultation_id)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "calculate-options":
        result = service.calculate_options(args.consultation_id)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "resume":
        result = service.resume_consultation(args.consultation_id, args.message)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "evaluate":
        from .mock_model import MockAgentModel
        from .requirements import TripRequirements
        from .workflow import run_requirements_graph

        result = run_requirements_graph(
            MockAgentModel(),
            TripRequirements(**consultation.trip_requirements).as_dict(),
            thread_id=consultation.thread_id,
            checkpoint_path=service.checkpoint_path,
        )
        print(json.dumps(result["decision"], ensure_ascii=False))
        return 0
    print(
        json.dumps(
            {
                "consultation_id": consultation.id,
                "thread_id": consultation.thread_id,
                "customer_name": consultation.customer_name,
                "status": consultation.status,
                "messages": [message.as_dict() for message in consultation.messages],
                "trip_requirements": consultation.trip_requirements,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
