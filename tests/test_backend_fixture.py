"""阶段 9 B2：双后端测试骨架。

B2 只交付「基础设施」，不交付「能跑的 Postgres 服务」——后者要等 B3 写完
``PostgresRepository``。所以这里验证的是：

1. ``--backend`` 默认 sqlite，缺省运行绝不依赖 Postgres；
2. ``postgres_schema`` 能建出独立 schema 并跑通 Alembic（10 张表）；
3. 测试结束后 schema 一定被清掉——哪怕测试失败；
4. ``search_path`` 的注入方式正确（不依赖数据库，纯字符串断言）。

第 2、3 条在没有 Postgres 时 **skip 而非失败**，保证默认 `pytest` 的
「455 passed」基线不被环境破坏。
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# 不需要 Postgres 的断言
# ---------------------------------------------------------------------------


def test_default_backend_option_is_sqlite():
    """默认值来自 ``pytest_addoption`` 的定义，而不是当前运行的取值。

    不能断言 ``pytestconfig.getoption("backend") == "sqlite"``——显式传
    ``--backend=postgres`` 时那条一定失败，而它想验证的其实是「没传时用什么」。
    """
    import conftest

    captured: dict[str, object] = {}

    class _Parser:
        def addoption(self, *args, **kwargs):
            captured.update(kwargs)

    conftest.pytest_addoption(_Parser())

    assert captured["default"] == "sqlite"
    assert set(captured["choices"]) == {"sqlite", "postgres"}


def test_search_path_is_injected_through_the_url():
    """Alembic 用 ``engine_from_config``，拿不到 ``connect_args``，
    只能靠连接串把迁移限定到某个 schema。
    """
    import conftest

    url = "postgresql+psycopg://u:p@127.0.0.1:5432/db"
    got = conftest._with_search_path(url, "test_abc123")

    assert got.startswith("postgresql+psycopg://u:p@127.0.0.1:5432/db?")
    # 不编码 '='：configparser 会把 '%' 当插值语法（见 conftest 的注释）
    assert "options=-csearch_path=test_abc123" in got


def test_search_path_appends_to_existing_query():
    import conftest

    url = "postgresql+psycopg://u:p@host/db?sslmode=disable"
    got = conftest._with_search_path(url, "test_x")

    assert "sslmode=disable" in got
    assert "&options=-csearch_path=test_x" in got


def test_postgres_schema_skips_without_backend_flag():
    """``--backend=sqlite`` 时 fixture 必须 skip，不能尝试连库。"""
    import conftest

    gen = conftest.postgres_schema.__wrapped__("sqlite")  # type: ignore[attr-defined]
    with pytest.raises(pytest.skip.Exception):
        next(gen)


def test_alembic_config_points_script_location_at_migrations():
    """配置的 script_location 必须指向项目的 migrations 目录。

    这条不连库：只检查路径解析，防止在 CI 里因工作目录不同而找不到迁移。
    """
    import conftest

    config = conftest._alembic_config(
        "postgresql+psycopg://u:p@127.0.0.1:5432/db", "test_abc"
    )

    assert config.get_main_option("script_location").endswith("migrations")
    assert "options=-csearch_path=test_abc" in config.get_main_option(
        "sqlalchemy.url"
    )


# ---------------------------------------------------------------------------
# 需要 Postgres 的断言
# ---------------------------------------------------------------------------


def test_schema_contains_every_migrated_table(postgres_connection):
    """fixture setup 之后，10 张业务表 + alembic_version 都在当前 schema 里。"""
    rows = postgres_connection.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = current_schema()"
    ).fetchall()
    tables = {row[0] for row in rows}

    assert "alembic_version" in tables
    assert {
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
    } <= tables


def test_each_test_gets_a_fresh_schema(postgres_connection):
    """fixture 每次生成的 schema 名唯一——隔离的全部意义就在这里。"""
    rows = postgres_connection.execute("SELECT current_schema()").fetchone()
    current = rows[0]

    assert current.startswith("test_")
    assert len(current) == len("test_") + 12


def test_drop_schema_removes_it(postgres_schema):
    """``drop_schema`` 确实清掉 schema 和里面的表。

    这里自己建一个临时 schema 来验证清理函数本身，而不是删掉 fixture 给的那个
    ——否则 teardown 会重复删除（幂等版能容忍，但断言会变得没有意义）。
    """
    import conftest

    url = conftest.postgres_url()
    assert url is not None
    admin = conftest._connect_postgres(url)
    probe = f"probe_{postgres_schema[5:]}"
    try:
        admin.execute(f"CREATE SCHEMA {probe}")
        assert admin.execute(
            "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
            (probe,),
        ).fetchone() is not None

        conftest.drop_schema(admin, probe)

        assert admin.execute(
            "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
            (probe,),
        ).fetchone() is None
    finally:
        conftest.drop_schema(admin, probe)
        admin.close()


def test_drop_schema_is_idempotent(postgres_schema):
    """重复清理不报错——否则测试失败后的 teardown 会掩盖真正的失败原因。"""
    import conftest

    url = conftest.postgres_url()
    assert url is not None
    admin = conftest._connect_postgres(url)
    try:
        conftest.drop_schema(admin, "definitely_not_an_existing_schema")
    finally:
        admin.close()
