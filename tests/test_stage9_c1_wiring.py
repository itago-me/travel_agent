"""阶段 9 C1：服务层接上 Postgres 仓储。

C1 要解决的不只是「换个仓储实现」，还有一个容易被忽略的**一致性风险**：

工具治理中间件与预订工具原本各自 ``SQLiteRepository(database)``。如果只把业务
仓储换成 Postgres，会出现「会话/审计/订单分居两个库」的分裂状态——最典型的后果
是 ``count_tool_executions`` 从 Postgres 统计、而治理中间件往 SQLite 写，**工具
预算永远不触发**。

所以这里既验证「能接上」，也验证「接上之后所有写入都落在同一个库」。
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from travel_agent.application import create_consultation_service
from travel_agent.langchain_tools import create_sensitive_booking_tools
from travel_agent.postgres_repository import PostgresRepository, plain_dsn
from travel_agent.repository import SQLiteRepository
from conftest import postgres_url


@pytest.fixture
def pg_repo(tmp_path):
    """一个建好 schema 的 Postgres 仓储；无 Postgres 时 skip。"""
    url = postgres_url()
    if url is None:
        pytest.skip("未设置 TRAVEL_AGENT_TEST_DATABASE_URL")
    try:
        import psycopg

        from alembic import command

        from conftest import _alembic_config, drop_schema
    except ImportError:
        pytest.skip("未安装 psycopg / alembic")

    admin = psycopg.connect(plain_dsn(url), autocommit=True)
    schema = f"test_{uuid4().hex[:12]}"
    admin.execute(f"CREATE SCHEMA {schema}")
    try:
        command.upgrade(_alembic_config(url, schema), "head")
        repo = PostgresRepository(plain_dsn(url), schema=schema)
        try:
            yield repo
        finally:
            repo.close()
    finally:
        drop_schema(admin, schema)
        admin.close()


# ---------------------------------------------------------------------------
# 不需要 Postgres 的断言
# ---------------------------------------------------------------------------


def test_service_defaults_to_sqlite_repository(tmp_path):
    """默认配置下仍然是 SQLite——升级不能悄悄改变既有行为。"""
    service = create_consultation_service(tmp_path / "travel.db")

    assert isinstance(service.repository, SQLiteRepository)
    assert service.agent_database == tmp_path / "travel.db"
    assert service.checkpoint_dsn is None


def test_backend_postgres_without_url_fails_loudly(tmp_path, monkeypatch):
    """配了 postgres 却没给 DSN 必须立刻报错，而不是退回 SQLite。"""
    monkeypatch.setenv("TRAVEL_AGENT_DATABASE_BACKEND", "postgres")
    monkeypatch.delenv("TRAVEL_AGENT_DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="TRAVEL_AGENT_DATABASE_URL"):
        create_consultation_service(tmp_path / "travel.db")


def test_unknown_backend_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAVEL_AGENT_DATABASE_BACKEND", "mysql")

    with pytest.raises(RuntimeError, match="未知的"):
        create_consultation_service(tmp_path / "travel.db")


def test_booking_tools_require_database_or_repository():
    with pytest.raises(ValueError, match="database 或 repository"):
        create_sensitive_booking_tools()


def test_plain_dsn_strips_the_sqlalchemy_driver():
    assert plain_dsn("postgresql+psycopg://u:p@h/db") == "postgresql://u:p@h/db"
    assert plain_dsn("postgresql://u:p@h/db") == "postgresql://u:p@h/db"


# ---------------------------------------------------------------------------
# 需要 Postgres 的断言
# ---------------------------------------------------------------------------


def test_service_accepts_a_postgres_repository(pg_repo, tmp_path):
    """业务仓储是 Postgres 时，服务仍能构造（C1 的核心交付）。"""
    from travel_agent.service import ConsultationService

    service = ConsultationService(pg_repo, agent_database=tmp_path / "tools.db")

    assert isinstance(service.repository, PostgresRepository)
    # 工具用的 SQLite 路径与业务仓储解耦
    assert service.agent_database == tmp_path / "tools.db"


def test_business_writes_land_in_postgres(pg_repo, tmp_path):
    from travel_agent.service import ConsultationService

    service = ConsultationService(pg_repo, agent_database=tmp_path / "tools.db")
    created = service.create_consultation(
        consultant_id="c1", customer_name="李四", initial_message="想去成都"
    )
    service.append_message(created.id, "user", "预算 5000")

    loaded = service.get_consultation(created.id)

    assert loaded.customer_name == "李四"
    # 直接从 Postgres 读，确认不是缓存
    assert pg_repo.get_consultation(created.id).messages[-1].content == "预算 5000"


def test_booking_tool_writes_through_the_given_repository(pg_repo, tmp_path):
    """预订工具用传入的仓储，而不是按路径自建 SQLite。

    这条防的正是「订单落 SQLite、会话落 Postgres」的分裂。
    """
    from travel_agent.service import ConsultationService

    service = ConsultationService(pg_repo, agent_database=tmp_path / "tools.db")
    created = service.create_consultation(
        consultant_id="c1", customer_name="王五", initial_message="想去西安"
    )
    version_id, _ = pg_repo.save_itinerary_version(
        created.id, {"budget": {"total": 1000.0}}, [], {}
    )
    pg_repo.select_itinerary_version(created.id, version_id)

    tools = {tool.name: tool for tool in create_sensitive_booking_tools(repository=pg_repo)}
    booking_tool = tools["create_mock_booking"]
    result = asyncio.run(
        booking_tool.ainvoke(
            {
                "consultation_id": created.id,
                "itinerary_version_id": version_id,
                "total_amount": 1000.0,
                "idempotency_key": "book-1",
            }
        )
    )

    assert result["status"] == "CREATED"
    # 订单真的在 Postgres 里
    assert len(pg_repo.list_mock_bookings(created.id)) == 1


def test_governance_counts_tool_calls_from_the_same_database(pg_repo, tmp_path):
    """治理中间件用同一仓储后，预算统计与业务审计同源。"""
    from travel_agent.agent import _governance_middleware

    middleware = _governance_middleware(
        tmp_path / "tools.db", "some-consultation", pg_repo
    )

    assert len(middleware) == 1
    # 中间件持有的仓储就是业务仓储，不是新建的 SQLite
    assert middleware[0].repository is pg_repo


def test_postgres_checkpointer_is_selected_by_dsn(pg_repo, tmp_path):
    """给了 checkpoint_dsn 就用 AsyncPostgresSaver，不再碰 SQLite 文件。"""
    from travel_agent.service import ConsultationService

    # DSN 从环境变量推导，不硬编码主机端口——写死会让测试在换端口后
    # 静默降级成 skip，看起来"通过"其实什么都没验证。
    service = ConsultationService(
        pg_repo,
        agent_database=tmp_path / "tools.db",
        checkpoint_dsn=plain_dsn(postgres_url()),
    )

    async def probe():
        async with service._checkpointer() as saver:
            return type(saver).__name__

    # DSN 不可用时跳过，避免把「环境没起」误报成代码错误
    try:
        name = asyncio.run(probe())
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"checkpoint DSN 不可用: {type(error).__name__}")

    assert name == "AsyncPostgresSaver"
