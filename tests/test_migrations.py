"""阶段 9 B1：Alembic schema 与 SQLite 手写迁移的一致性。

两套 schema 并存是本阶段已知的代价：SQLite 走 ``repository._initialize()``
的手写迁移，Postgres 走 Alembic。**它们会不会漂移，只能靠断言。**

这个文件做三件事：

1. 从 SQLite 侧读出真实的表/列/索引，作为「参考实现」；
2. 让 Alembic 在 Postgres 上建出结构，再读回来对比；
3. Postgres 不可用时整组跳过——**默认 `pytest` 必须不需要 Postgres 就能全绿**。

第 3 点很重要：如果这里因为环境没起来而失败，会让「451 个测试全绿」这条
基线变得不可信，而那条基线是本阶段「随时可停」的保证。
"""

from __future__ import annotations

import os

import pytest
from uuid import uuid4
import sqlite3

from travel_agent.repository import SQLiteRepository


# 参考实现：SQLite 手写迁移建出的表。
EXPECTED_TABLES = {
    "consultations",
    "messages",
    "audit_events",
    "turn_spans",
    "itinerary_drafts",
    "itinerary_versions",
    "provider_requests",
    "customer_preferences",
    "approval_requests",
    "mock_bookings",
}


@pytest.fixture(scope="module")
def sqlite_schema(tmp_path_factory):
    """用 SQLite 手写迁移建一个库，读出它的结构作为比对基准。"""
    database = tmp_path_factory.mktemp("schema") / "reference.db"
    SQLiteRepository(database)

    connection = sqlite3.connect(database)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            if not row[0].startswith("sqlite_")
        }
        columns = {
            table: {
                row[1]
                for row in connection.execute(f"PRAGMA table_info({table})")
            }
            for table in tables
        }
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
            )
        }
    finally:
        connection.close()
    return {"tables": tables, "columns": columns, "indexes": indexes}


# ---------------------------------------------------------------------------
# 不需要 Postgres 的断言
# ---------------------------------------------------------------------------


def test_sqlite_reference_has_every_expected_table(sqlite_schema):
    """基准本身要正确——否则后面「对比通过」毫无意义。"""
    assert sqlite_schema["tables"] == EXPECTED_TABLES


def _load_migration_statements() -> list[str]:
    """按路径加载 0001 迁移模块，取出它要执行的 SQL 列表。

    用 importlib 而不是 ``import migrations.versions.xxx``：``migrations/`` 不是
    包（没有 ``__init__.py``，也不该有——它由 Alembic 按路径加载），加包结构
    反而会让 Alembic 的脚本发现逻辑和 Python 的导入机制互相干扰。
    """
    import importlib.util
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "migrations"
        / "versions"
        / "0001_initial_schema.py"
    )
    spec = importlib.util.spec_from_file_location("migration_0001", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module._UPGRADE_STATEMENTS)


def test_migration_script_covers_every_table():
    """迁移脚本里的建表语句必须覆盖全部 10 张表。

    这条不连库——直接读迁移模块里的 SQL 常量，所以永远能跑。
    """
    created = set()
    for statement in _load_migration_statements():
        normalized = " ".join(statement.split())
        if normalized.startswith("CREATE TABLE"):
            created.add(normalized.split()[2])
    assert created == EXPECTED_TABLES


def test_migration_defines_both_indexes():
    created = [
        " ".join(s.split())
        for s in _load_migration_statements()
        if " ".join(s.split()).startswith(("CREATE INDEX", "CREATE UNIQUE INDEX"))
    ]
    # turn_spans 的普通索引 + approval_requests 的部分唯一索引
    assert len(created) == 2


def test_migration_keeps_check_constraints():
    """6 处 CHECK 是标准 SQL，迁移里必须一个不少地保留。"""
    joined = " ".join(" ".join(s.split()) for s in _load_migration_statements())

    assert joined.count("CHECK (status IN") == 2
    assert joined.count("CHECK (decision IS NULL OR decision IN") == 1


# ---------------------------------------------------------------------------
# 需要 Postgres 的断言
# ---------------------------------------------------------------------------


def _postgres_url() -> str | None:
    url = os.getenv("TRAVEL_AGENT_TEST_DATABASE_URL", "").strip()
    return url or None


@pytest.fixture
def postgres_ready():
    """连不上 Postgres 就跳过，不让整套测试失败。"""
    url = _postgres_url()
    if not url:
        pytest.skip("未设置 TRAVEL_AGENT_TEST_DATABASE_URL")
    try:
        import psycopg

        from conftest import plain_url
    except ImportError:
        pytest.skip("未安装 psycopg")

    try:
        with psycopg.connect(plain_url(url), connect_timeout=3):
            pass
    except Exception as error:  # noqa: BLE001 - 任何连接失败都按「不可用」处理
        pytest.skip(f"Postgres 不可用: {type(error).__name__}")
    return url


@pytest.fixture
def alembic_schema(postgres_ready):
    """在独立 schema 里跑 Alembic，结束清理。

    早期版本直接跑在 ``public`` 上，结果和 LangGraph 的 checkpoint 表
    （``checkpoints`` / ``checkpoint_writes`` 等）互相干扰——断言「建出的表
    等于 10 张业务表」会因为 public 里还有别人的表而失败。schema 隔离后，
    「这个 schema 里有什么」就是没有歧义的问题。
    """
    import psycopg

    from alembic import command

    from conftest import _alembic_config as make_config
    from conftest import drop_schema, plain_url

    url = postgres_ready
    schema = f"mig_{uuid4().hex[:12]}"
    admin = psycopg.connect(plain_url(url), autocommit=True)
    admin.execute(f"CREATE SCHEMA {schema}")
    config = make_config(url, schema)
    command.upgrade(config, "head")
    try:
        yield schema, config
    finally:
        drop_schema(admin, schema)
        admin.close()


def _tables_in(schema: str) -> set[str]:
    """Tables currently present in ``schema``."""
    import psycopg

    from conftest import plain_url

    with psycopg.connect(plain_url(_postgres_url())) as connection:
        rows = connection.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = %s", (schema,)
        ).fetchall()
    return {row[0] for row in rows}


def _columns_in(schema: str, table: str) -> set[str]:
    import psycopg

    from conftest import plain_url

    with psycopg.connect(plain_url(_postgres_url())) as connection:
        rows = connection.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s",
            (schema, table),
        ).fetchall()
    return {row[0] for row in rows}


def test_upgrade_creates_the_same_tables_as_sqlite(alembic_schema, sqlite_schema):
    """Alembic 建出的表集合必须与 SQLite 参考实现一致。"""
    schema, _ = alembic_schema
    created = _tables_in(schema) - {"alembic_version"}

    assert created == sqlite_schema["tables"], (
        f"多出：{created - sqlite_schema['tables']}，"
        f"缺失：{sqlite_schema['tables'] - created}"
    )


def test_upgrade_creates_the_same_columns_as_sqlite(alembic_schema, sqlite_schema):
    """逐表比对列名——漏一列就是 B3 里某个方法会炸。"""
    schema, _ = alembic_schema

    for table, expected_columns in sqlite_schema["columns"].items():
        actual = _columns_in(schema, table)
        assert actual == expected_columns, (
            f"{table} 多出：{actual - expected_columns}，"
            f"缺失：{expected_columns - actual}"
        )


def test_downgrade_removes_everything(alembic_schema):
    """downgrade base 之后业务表必须清空（alembic_version 会保留）。"""
    from alembic import command

    schema, config = alembic_schema
    command.downgrade(config, "base")

    remaining = _tables_in(schema) - {"alembic_version"}

    assert remaining == set(), f"残留表：{sorted(remaining)}"
