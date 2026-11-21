"""应用装配：根据配置决定业务数据落在 SQLite 还是 Postgres。

## 三个开关

``TRAVEL_AGENT_DATABASE_BACKEND``（业务仓储）
    ``sqlite``（默认）| ``postgres``。显式开关而非自动探测——
    「以为连的是 Postgres 结果写进了本地文件」是迁移期最容易出的错。

``TRAVEL_AGENT_DATABASE_URL``（业务数据的连接串）
    只在 ``backend=postgres`` 时需要。**同时被 Alembic 用来建表**，
    所以它既是运行时配置也是迁移配置。

``TRAVEL_AGENT_CHECKPOINT_DSN``（对话状态存哪，可选）
    不设时**跟随业务后端**：业务在 Postgres，checkpoint 也在 Postgres。
    设了就用它，用来把状态库单独放一个实例。

## 为什么 checkpoint 要跟随后端

早期版本让两者各管各的，结果是「设了 backend=postgres 之后，业务数据进了
Postgres，对话状态还在 localhost 的 sqlite 文件里」。这在容器部署下会静默丢
数据：那个 .sqlite 文件没挂卷，容器一重建，所有会话历史消失，用户发现「对话
接不下去」却没有任何报错。

## 关于 .env

模块导入时调用一次 ``load_dotenv(override=False)``。**导入时而不是使用时**，
是因为这三个变量在服务构造阶段就被读取，而那时通常还没走到
``load_agent_settings()``（它才是原先唯一加载 .env 的地方）。早先的表现是：
写进 ``.env`` 的后端配置被静默忽略，命令行里看不见任何提示。

``override=False`` 让真实环境变量优先于 ``.env``——临时覆盖配置文件是符合
直觉的优先级。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .repository import SQLiteRepository
from .service import ConsultationService

# 导入时加载：见模块文档。override=False 保证 export 的值优先。
load_dotenv(override=False)


def checkpoint_database_path(database_path: str | Path) -> Path:
    database_path = Path(database_path)
    return database_path.with_name(f"{database_path.stem}_checkpoints.sqlite")


def database_backend() -> str:
    """Which backend the business repository should use.

    Defaults to sqlite so an existing checkout keeps working with no
    configuration; ``postgres`` requires ``TRAVEL_AGENT_DATABASE_URL``.
    """
    return (
        os.getenv("TRAVEL_AGENT_DATABASE_BACKEND", "sqlite").strip().lower() or "sqlite"
    )


def database_url() -> str:
    """The business database DSN.

    Raises rather than falling back to SQLite: silently writing to a file when
    Postgres was requested is the failure mode this whole module guards against.
    """
    dsn = os.getenv("TRAVEL_AGENT_DATABASE_URL", "").strip()
    if not dsn:
        raise RuntimeError(
            "TRAVEL_AGENT_DATABASE_BACKEND=postgres 需要同时设置 "
            "TRAVEL_AGENT_DATABASE_URL，例如：\n"
            "  export TRAVEL_AGENT_DATABASE_URL="
            "'postgresql+psycopg://user:pass@127.0.0.1:5432/travel_agent'"
        )
    return dsn


def checkpoint_dsn() -> str | None:
    """Where LangGraph should keep conversation state, or ``None`` for SQLite.

    Explicit ``TRAVEL_AGENT_CHECKPOINT_DSN`` wins; otherwise the checkpoint
    follows the business backend so that "backend=postgres" means "everything
    in postgres" and cannot half-migrate.
    """
    explicit = os.getenv("TRAVEL_AGENT_CHECKPOINT_DSN", "").strip()
    if explicit:
        return explicit
    if database_backend() == "postgres":
        return database_url()
    return None


def create_business_repository(database_path: str | Path):
    """Build the repository the business data lives in."""
    backend = database_backend()
    if backend == "sqlite":
        return SQLiteRepository(database_path)
    if backend == "postgres":
        from .postgres_repository import PostgresRepository, plain_dsn

        return PostgresRepository(plain_dsn(database_url()))
    raise RuntimeError(f"未知的 TRAVEL_AGENT_DATABASE_BACKEND: {backend}")


def create_consultation_service(
    database_path: str | Path,
    checkpoint_path: str | Path | None = None,
) -> ConsultationService:
    """Build the application service used by every entry point."""
    repository = create_business_repository(database_path)

    agent_database = database_path

    return ConsultationService(
        repository,
        # checkpoint_dsn 存在时这个路径不会被用到，但仍然算出来——
        # 留一个确定的默认值比留 None 更容易排查「到底连了哪个库」。
        checkpoint_path=checkpoint_path or checkpoint_database_path(agent_database),
        agent_database=agent_database,
        checkpoint_dsn=checkpoint_dsn(),
    )
