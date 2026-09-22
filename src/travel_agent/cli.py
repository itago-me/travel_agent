import argparse
import json
from pathlib import Path

from .application import create_consultation_service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="travel-agent")
    parser.add_argument("--database", default="data/travel_agent.db")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--consultant-id", required=True)
    start.add_argument("--customer-name", required=True)
    start.add_argument("--message", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("consultation_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = create_consultation_service(Path(args.database))
    if args.command == "start":
        consultation = service.create_consultation(
            args.consultant_id, args.customer_name, args.message
        )
        print(json.dumps({"consultation_id": consultation.id, "thread_id": consultation.thread_id}))
        return 0
    consultation = service.get_consultation(args.consultation_id)
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
