"""常驻服务 HTTP 入口（阶段 6.4）。

目标形态（schedule 6.4）：一个进程、一个 event loop、一个复用的
ConsultationService，对外提供完整业务闭环的 HTTP 接口。CLI 与 HTTP
共用同一 service 层，不各自实现业务逻辑。

端点与 CLI 子命令一一对应（见 schedule 6.4 第 2 节），便于两边对照。

启动：
    PYTHONPATH=src:.deps python -m travel_agent.api --database data/travel_agent.db
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .application import create_consultation_service
from .persistent import PersistentLoopRunner


class ChatRequest(BaseModel):
    message: str
    model_name: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    env_file: str | None = None


class StartRequest(BaseModel):
    consultant_id: str
    customer_name: str
    message: str
    customer_id: str | None = None


class ApprovalDecisionRequest(BaseModel):
    decided_by: str = "cli"
    reason: str = ""
    edited_arguments: dict[str, Any] | None = None


def _resolve_api_token(explicit: str | None = None) -> str | None:
    """Token from the argument or the environment; ``None`` means auth off.

    Auth is opt-in so local development stays frictionless, but an unset token
    is loud at startup - see :func:`main`. Anything reachable from a network
    must set ``TRAVEL_AGENT_API_TOKEN``.
    """
    token = explicit if explicit is not None else os.getenv("TRAVEL_AGENT_API_TOKEN")
    return token or None


def create_app(database_path: str | Path, api_token: str | None = None) -> FastAPI:
    """Build the FastAPI app around one shared service instance.

    ``api_token`` guards every business endpoint with a bearer token. When it
    is ``None`` the API is unauthenticated, which is only appropriate on
    localhost - an unauthenticated endpoint can spend real model budget.
    """
    service = create_consultation_service(database_path)
    token = _resolve_api_token(api_token)
    app = FastAPI(title="travel-agent", version="6.4")

    def require_token(authorization: str | None = Header(default=None)) -> None:
        """Reject a request that does not carry the expected bearer token."""
        if token is None:
            return
        expected = f"Bearer {token}"
        # 常量时间比较：普通 != 会按字节短路，泄露 token 前缀。
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(
                status_code=401,
                detail="missing or invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # 业务端点统一挂这个依赖；文档与健康检查保持开放。
    auth = [Depends(require_token)]

    @app.exception_handler(KeyError)
    async def _not_found_handler(_request, exc: KeyError) -> JSONResponse:
        """Translate repository KeyErrors into 404.

        领域层用 ``KeyError`` 表示"这个 id 不存在"（见 repository.py）。
        不翻译的话 FastAPI 会返回 500，调用方会以为服务崩了，而实际上
        只是一个查无此物——状态码的语义在这里必须正确。
        """
        detail = exc.args[0] if exc.args else "not found"
        return JSONResponse(status_code=404, content={"detail": str(detail)})

    def _require_consultation(consultation_id: str):
        """Fetch a consultation or raise 404 - used by every id-scoped route."""
        consultation = _find_consultation(service, consultation_id)
        if consultation is None:
            raise HTTPException(status_code=404, detail="consultation not found")
        return consultation

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Unauthenticated liveness probe.

        Deliberately outside the auth dependency: a load balancer must be able
        to check liveness without holding a secret, and this endpoint exposes
        no business data.
        """
        return {"status": "ok", "auth": "bearer" if token else "none"}

    @app.post("/consultations", dependencies=auth)
    def start_consultation(request: StartRequest) -> dict[str, Any]:
        consultation = service.create_consultation(
            request.consultant_id,
            request.customer_name,
            request.message,
            customer_id=request.customer_id,
        )
        return {
            "consultation_id": consultation.id,
            "thread_id": consultation.thread_id,
            "status": consultation.status,
            "trip_requirements": consultation.trip_requirements,
        }

    @app.get("/consultations/{consultation_id}", dependencies=auth)
    def get_consultation(consultation_id: str) -> dict[str, Any]:
        consultation = _require_consultation(consultation_id)
        # Consultation 是可变聚合根，没有 as_dict（与 CLI status 命令输出保持一致）
        return {
            "consultation_id": consultation.id,
            "thread_id": consultation.thread_id,
            "customer_id": consultation.customer_id,
            "customer_name": consultation.customer_name,
            "status": consultation.status,
            "trip_requirements": consultation.trip_requirements,
            "planning_constraints": consultation.planning_constraints,
            "messages": [message.as_dict() for message in consultation.messages],
            "created_at": consultation.created_at,
            "updated_at": consultation.updated_at,
        }

    @app.post("/consultations/{consultation_id}/chat", dependencies=auth)
    def chat(consultation_id: str, request: ChatRequest) -> dict[str, Any]:
        _require_consultation(consultation_id)
        return service.chat_consultation(
            consultation_id,
            request.message,
            model_name=request.model_name,
            api_key=request.api_key,
            base_url=request.base_url,
            env_file=request.env_file,
        )

    @app.get("/consultations/{consultation_id}/itineraries", dependencies=auth)
    def itineraries(consultation_id: str) -> list[dict[str, Any]]:
        _require_consultation(consultation_id)
        return service.repository.list_itinerary_versions(consultation_id)

    @app.post("/consultations/{consultation_id}/itineraries/{version_id}/select", dependencies=auth)
    def select_itinerary(consultation_id: str, version_id: str) -> dict[str, Any]:
        return service.select_itinerary(consultation_id, version_id)

    @app.post("/consultations/{consultation_id}/chat/stream", dependencies=auth)
    def chat_stream(consultation_id: str, request: ChatRequest) -> StreamingResponse:
        """Server-sent events for one agent turn.

        The blocking sync generator from the service layer is drained into SSE
        frames. ``X-Accel-Buffering: no`` stops reverse proxies from buffering
        the stream into uselessness - without it the client sees everything at
        once and streaming buys nothing.
        """
        _require_consultation(consultation_id)

        def frames():
            try:
                for event in service.chat_consultation_stream(
                    consultation_id,
                    request.message,
                    model_name=request.model_name,
                    api_key=request.api_key,
                    base_url=request.base_url,
                    env_file=request.env_file,
                ):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as error:  # noqa: BLE001 - 必须让客户端知道为什么断了
                payload = {"type": "error", "message": str(error)}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/consultations/{consultation_id}/usage", dependencies=auth)
    def usage(consultation_id: str) -> dict[str, Any]:
        """Token spend per consultation - the bill-shaped view of a conversation."""
        _require_consultation(consultation_id)
        return service.consultation_usage(consultation_id)

    @app.get("/consultations/{consultation_id}/approvals", dependencies=auth)
    def pending_approvals(consultation_id: str) -> list[dict[str, Any]]:
        return [
            approval.as_dict()
            for approval in service.list_pending_approvals(consultation_id)
        ]

    @app.post("/approvals/{approval_id}/approve", dependencies=auth)
    def approve(approval_id: str, request: ApprovalDecisionRequest) -> dict[str, Any]:
        return service.decide_approval(
            approval_id,
            "APPROVE",
            decided_by=request.decided_by,
            reason=request.reason,
            edited_arguments=request.edited_arguments,
        )

    @app.post("/approvals/{approval_id}/reject", dependencies=auth)
    def reject(approval_id: str, request: ApprovalDecisionRequest) -> dict[str, Any]:
        return service.decide_approval(
            approval_id,
            "REJECT",
            decided_by=request.decided_by,
            reason=request.reason,
        )

    @app.post("/consultations/{consultation_id}/resume", dependencies=auth)
    def resume(consultation_id: str) -> dict[str, Any]:
        # 恢复不接受外部 interrupt_id：service 按 thread 上未决的审批自动续跑
        return service.resume_approved_consultation(consultation_id)

    @app.get("/customers/{customer_id}/preferences", dependencies=auth)
    def preferences(customer_id: str) -> list[dict[str, Any]]:
        return [
            preference.as_dict()
            for preference in service.list_customer_preferences(customer_id)
        ]

    @app.put("/customers/{customer_id}/preferences/{key}", dependencies=auth)
    def set_preference(customer_id: str, key: str, body: dict[str, Any]) -> dict[str, Any]:
        service.set_customer_preference(
            customer_id, key, body.get("value"), source=body.get("source", "api")
        )
        return {"ok": True}

    @app.delete("/customers/{customer_id}/preferences/{key}", dependencies=auth)
    def delete_preference(customer_id: str, key: str) -> dict[str, Any]:
        removed = service.delete_customer_preference(customer_id, key)
        if not removed:
            raise HTTPException(status_code=404, detail="preference not found")
        return {"ok": True}

    return app


def _default_database() -> str:
    """Database the module-level app binds to.

    Read from the environment so the standard uvicorn invocation works without
    an argument: ``uvicorn travel_agent.api:app``. The CLI entry point
    (``python -m travel_agent.api``) still takes ``--database`` explicitly.
    """
    import os

    return os.getenv("TRAVEL_AGENT_DATABASE", "data/travel_agent.db")


# ASGI 入口。uvicorn 约定通过 "module:attribute" 定位应用，工厂函数
# create_app() 不是可调用对象，因此必须暴露一个模块级的 app。
app = create_app(_default_database(), _resolve_api_token())


def _find_consultation(service, consultation_id: str):
    from .domain import Consultation

    try:
        return service.get_consultation(consultation_id)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(prog="travel-agent-api")
    parser.add_argument("--database", default="data/travel_agent.db")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument(
        "--api-token",
        default=None,
        help="bearer token; defaults to $TRAVEL_AGENT_API_TOKEN, else auth is off",
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is required for the persistent service.\n"
            "install it with: pip install uvicorn"
        )
        return 2

    import sys

    token = _resolve_api_token(args.api_token)
    app = create_app(args.database, token)
    print(f"travel-agent persistent service on http://{args.host}:{args.port}")
    if token:
        print("auth: bearer token required on every business endpoint")
    else:
        print(
            "auth: DISABLED - every endpoint is open. Set TRAVEL_AGENT_API_TOKEN "
            "before exposing this service beyond localhost."
        )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
