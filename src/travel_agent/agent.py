"""Configuration and factory for the production LangChain Agent."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .flyai_client import FlyAIClient
from .provider_factory import ProviderSettings


@dataclass(frozen=True)
class AgentSettings:
    model_mode: str = "openai_compatible"
    model_name: str | None = None
    api_key: str | None = None
    base_url: str | None = None


def _load_dotenv(env_file: str | Path | None = None) -> None:
    try:
        dotenv = importlib.import_module("dotenv")
    except ImportError:
        return
    dotenv.load_dotenv(dotenv_path=env_file or ".env", override=False)


def load_agent_settings(env_file: str | Path | None = None) -> AgentSettings:
    """Load model configuration without ever placing secrets in source code."""
    _load_dotenv(env_file)
    return AgentSettings(
        model_mode=os.getenv("TRAVEL_AGENT_MODEL_MODE", "openai_compatible"),
        model_name=os.getenv("TRAVEL_AGENT_MODEL_NAME"),
        api_key=os.getenv("TRAVEL_AGENT_API_KEY"),
        base_url=os.getenv("TRAVEL_AGENT_BASE_URL"),
    )


def _import_chat_openai():
    try:
        return importlib.import_module("langchain_openai").ChatOpenAI
    except ImportError:
        return None


def _import_create_agent():
    try:
        return importlib.import_module("langchain.agents").create_agent
    except ImportError:
        return None


def create_chat_model(settings: AgentSettings):
    """Create the configured chat model for the Agent or summary worker."""
    chat_openai = _import_chat_openai()
    if chat_openai is None:
        raise RuntimeError("请先安装 langchain-openai，才能使用 Chat Model。")
    if not settings.model_name:
        raise ValueError("缺少 TRAVEL_AGENT_MODEL_NAME，请配置 Chat Model 名称。")
    model_kwargs: dict[str, Any] = {"model": settings.model_name}
    if settings.api_key:
        model_kwargs["api_key"] = settings.api_key
    if settings.base_url:
        model_kwargs["base_url"] = settings.base_url
    return chat_openai(**model_kwargs)


def _positive_env(name: str, default: int) -> int:
    """Read a positive int from the environment, falling back on bad input."""
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _governance_middleware(
    database: str | Path, consultation_id: str | None
) -> list[Any]:
    """Tool audit + call budget, only when the turn belongs to a consultation.

    ``consultation_id`` is absent for factory smoke tests that only assert tool
    wiring, so governance stays out of that path rather than inventing an owner.
    """
    if not consultation_id:
        return []
    from .repository import SQLiteRepository
    from .tool_governance import ToolGovernanceMiddleware

    return [
        ToolGovernanceMiddleware(
            SQLiteRepository(database),
            consultation_id,
            max_calls=_positive_env("TRAVEL_AGENT_MAX_TOOL_CALLS", 30),
        )
    ]


def create_agent_app(
    database: str | Path,
    settings: AgentSettings | None = None,
    *,
    checkpointer: Any | None = None,
    provider_settings: ProviderSettings | None = None,
    flyai_client: FlyAIClient | None = None,
    tools: list[Any] | None = None,
    consultation_id: str | None = None,
):
    """Create the production Agent using LangChain's official factory."""
    settings = settings or load_agent_settings()
    if settings.model_mode != "openai_compatible":
        raise ValueError(
            "阶段 3.2 只支持 openai_compatible 模式；请设置 TRAVEL_AGENT_MODEL_MODE。"
        )
    if not settings.model_name:
        raise ValueError(
            "缺少 TRAVEL_AGENT_MODEL_NAME，请在 .env 中配置真实 Chat Model 名称。"
        )
    create_agent = _import_create_agent()
    if create_agent is None:
        raise RuntimeError("请先安装完整 langchain，才能使用 create_agent。")
    model = create_chat_model(settings)
    from langchain.agents.middleware import HumanInTheLoopMiddleware

    from .langchain_tools import create_sensitive_booking_tools, create_travel_tools
    from .agent_context import (
        ContextWindowMiddleware,
        TravelMemoryMiddleware,
        TravelAgentState,
    )

    provider_settings = provider_settings or ProviderSettings.from_env()
    if tools is None:
        tools = [
            *create_travel_tools(
                database,
                provider_settings=provider_settings,
                flyai_client=flyai_client,
            ),
            *create_sensitive_booking_tools(database),
        ]
    agent_kwargs: dict[str, Any] = {
        "model": model,
        "tools": tools,
        "state_schema": TravelAgentState,
        "middleware": [
            TravelMemoryMiddleware(
                model,
                max_messages=_positive_env("TRAVEL_AGENT_CONTEXT_MAX_MESSAGES", 12),
                max_tokens=_positive_env("TRAVEL_AGENT_CONTEXT_MAX_TOKENS", 4000),
            ),
            ContextWindowMiddleware(
                max_messages=_positive_env("TRAVEL_AGENT_CONTEXT_MAX_MESSAGES", 12),
                max_tokens=_positive_env("TRAVEL_AGENT_CONTEXT_MAX_TOKENS", 4000),
            ),
            *_governance_middleware(database, consultation_id),
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "create_mock_booking": {
                        "allowed_decisions": ["approve", "edit", "reject"]
                    },
                    "cancel_mock_booking": {
                        "allowed_decisions": ["approve", "edit", "reject"]
                    },
                },
                description_prefix="敏感旅行操作需要顾问审批",
            ),
        ],
        "system_prompt": (
            "你是中国境内旅行顾问 Copilot。"
            "先理解顾问需求，必要时调用旅行查询工具；"
            "根据工具返回的结构化结果，用中文给出清晰、可核验的建议。"
            "仅在顾问明确要求时申请创建或取消模拟预订；"
            "创建前必须有已选择的方案版本，不得编造咨询、方案或订单 ID。"
            "敏感工具会暂停等待人工审批，提出申请不代表订单已经创建或取消。"
        ),
    }
    if checkpointer is not None:
        agent_kwargs["checkpointer"] = checkpointer
    return create_agent(**agent_kwargs)
