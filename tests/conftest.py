"""Shared test setup: backend selection, schema isolation, ambient-config hygiene.

三件事：

1. **隔离 ``.env``**：``load_agent_settings`` 调用 ``dotenv.load_dotenv``，会把
   文件里的值写进 ``os.environ`` 且终生不清。一个开发者的 ``.env`` 因此能改变
   后续无关测试的行为——曾出现「``.env`` 里有 API_TOKEN 导致后面所有 API 测试
   变 401」，失败点指向 auth 而不是真正触发加载的那个测试。
2. **``--backend`` 开关**：默认 ``sqlite``，保证「不装 Postgres 也能跑全套」。
   这是阶段 9 的硬约束，不是偏好——那条基线让每一步都能随时停下。
3. **``postgres_schema`` fixture**：为单个测试建一个独立的 Postgres schema，
   跑 Alembic 建表，测试结束 ``DROP SCHEMA ... CASCADE``。

为什么用 schema 隔离而不是 TRUNCATE：TRUNCATE 要维护一张表清单，新增表忘了加
就会让测试互相污染；``DROP SCHEMA CASCADE`` 不依赖清单，且天然支持并行。
"""

from __future__ import annotations

import os
from urllib.parse import quote, urlsplit, urlunsplit
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# 收集阶段就要中立化：这一步必须在 import 时做，不能用 fixture。
#
# ``travel_agent.api`` 在模块级构造 ``app = create_app(...)``，所以「导入 API
# 测试模块」就等于「按当前环境建一次服务」。若开发机的 .env 写了
# ``TRAVEL_AGENT_DATABASE_BACKEND=postgres`` 却漏了 URL，**收集阶段**就抛
# RuntimeError，pytest 连一个用例都跑不起来——fixture 再晚都来不及。
#
# 固定成 sqlite，让「跑测试」与「开发机怎么配数据库」彻底无关。
# ---------------------------------------------------------------------------
os.environ["TRAVEL_AGENT_DATABASE_BACKEND"] = "sqlite"


# ---------------------------------------------------------------------------
# 隔离 .env 带来的全局污染
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_from_dotenv(monkeypatch):
    """把「会改变行为」的变量钉死，但不夺走测试真正需要的配置。

    这里做过两次方向修正，原因值得记下来：

    1. 最初只清 ``TRAVEL_AGENT_API_TOKEN``。后来 ``application.py`` 开始在导入时
       加载 ``.env``，于是 ``.env`` 里的 ``TRAVEL_AGENT_*_PROVIDER=flyai`` 真的进入
       了环境，本该用 sqlite provider 的用例开始调外部命令——4 个失败。
    2. 于是改成「清掉所有 ``TRAVEL_AGENT_*``」。但 ``load_agent_settings()`` 会
       重新加载 ``.env``，把模型名之类又读回来；一刀切地删会让依赖
       ``TRAVEL_AGENT_MODEL_NAME`` 的用例报「缺少模型名」——14 个失败。

    最终做法：**只钉住会改变路由的变量**（数据库后端与 provider 选择），其余保留。
    用 ``setenv`` 而不是 ``delenv`` 很关键——``load_dotenv(override=False)`` 不会
    覆盖已存在的键，所以 .env 里再写一次也改不动它们。
    """
    # 数据库：测试永远跑 SQLite，与开发机有没有配 Postgres 无关。
    monkeypatch.setenv("TRAVEL_AGENT_DATABASE_BACKEND", "sqlite")
    monkeypatch.delenv("TRAVEL_AGENT_DATABASE_URL", raising=False)
    monkeypatch.delenv("TRAVEL_AGENT_CHECKPOINT_DSN", raising=False)

    # provider：默认走本地 sqlite 数据源，不打外部命令。
    for role in ("HOTEL", "TRANSPORT", "ATTRACTION"):
        monkeypatch.setenv(f"TRAVEL_AGENT_{role}_PROVIDER", "sqlite")

    # 认证：保持原有的隔离意图（.env 里有 token 会让 API 测试变成 401）。
    monkeypatch.delenv("TRAVEL_AGENT_API_TOKEN", raising=False)


# ---------------------------------------------------------------------------
# 后端选择
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        default="sqlite",
        choices=("sqlite", "postgres"),
        help="repository backend fixtures should target (default: sqlite)",
    )


@pytest.fixture(scope="session")
def backend(request) -> str:
    """Which backend this run targets."""
    return request.config.getoption("--backend")


#: 测试用的 Postgres 连接串，刻意与应用的 ``TRAVEL_AGENT_DATABASE_URL`` 分开。
#:
#: 用同一个变量意味着「开发机 .env 里配了生产库」时，测试会直接连过去跑
#: ``DROP SCHEMA``——代价太大。分名之后，只有显式设了测试变量才会跑 PG 用例。
TEST_DATABASE_URL_ENV = "TRAVEL_AGENT_TEST_DATABASE_URL"

def postgres_url() -> str | None:
    """The configured test Postgres URL, or ``None`` when unavailable.

    Kept as a plain function (not a fixture) so other test modules can reuse the
    same availability logic without importing fixture plumbing.
    """
    if os.getenv("TRAVEL_AGENT_TEST_SKIP_POSTGRES", "").strip() in {"1", "true", "yes"}:
        return None
    return os.getenv(TEST_DATABASE_URL_ENV, "").strip() or None


def plain_url(url: str) -> str:
    """Strip the SQLAlchemy driver suffix: ``postgresql+psycopg://`` → ``postgresql://``.

    同一个数据库有两种写法：Alembic/SQLAlchemy 需要 ``+psycopg`` 来选方言，
    而 ``psycopg.connect()`` 说的是 libpq 语法，遇到 ``+`` 直接报
    ``missing "=" after ...``（实测踩过）。
    """
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.split("+")[0], parts.netloc, parts.path, parts.query, parts.fragment))


def _connect_postgres(url: str):
    """Open a connection, converting any failure into a skip.

    ``psycopg`` is imported lazily: the default sqlite run must not require the
    Postgres driver to be installed.
    """
    try:
        import psycopg
    except ImportError:
        pytest.skip("未安装 psycopg（B3 阶段才需要）")

    try:
        return psycopg.connect(plain_url(url), connect_timeout=3, autocommit=True)
    except Exception as error:  # noqa: BLE001 - 任何连接失败都按「不可用」处理
        pytest.skip(f"Postgres 不可用: {type(error).__name__}: {error}")


def _with_search_path(url: str, schema: str) -> str:
    """Return ``url`` with a libpq ``options=-csearch_path=<schema>`` appended.

    Alembic uses ``engine_from_config`` and therefore cannot be handed
    ``connect_args``; the only way to confine it to one schema is through the
    connection string.

    **不要把 ``=`` 编码成 ``%3D``**：这个字符串最终会经
    ``config.set_main_option`` 写进 configparser，而 ``%`` 在 INI 里是插值语法，
    会直接抛 ``ValueError: invalid interpolation syntax``（实测踩过）。
    ``=`` 在 URL query 的取值位置本身就是合法字符，SQLAlchemy 会按第一个 ``=``
    切分，得到 ``options=-csearch_path=<schema>``。
    """
    parts = urlsplit(url)
    extra = f"options=-csearch_path={quote(schema, safe='')}"
    query = f"{parts.query}&{extra}" if parts.query else extra
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


@pytest.fixture
def postgres_schema(backend):
    """An empty Postgres schema with the Alembic schema applied.

    Yields the schema name; drops it (and everything in it) on teardown, even
    when the test fails.
    """
    if backend != "postgres":
        pytest.skip("需要 --backend=postgres")

    url = postgres_url()
    if url is None:
        pytest.skip(f"未设置 {TEST_DATABASE_URL_ENV}")

    admin = _connect_postgres(url)
    schema = f"test_{uuid4().hex[:12]}"
    admin.execute(f"CREATE SCHEMA {schema}")
    try:
        _run_alembic(url, schema, "head")
        yield schema
    finally:
        drop_schema(admin, schema)
        admin.close()


def drop_schema(admin, schema: str) -> None:
    """Drop a test schema, tolerating an already-gone schema.

    幂等是有意的：断言「清理逻辑有效」的测试会自己先删一次，如果这里要求
    schema 必须存在，那条测试就会在 teardown 阶段报错（实测踩过）。
    """
    admin.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


def _alembic_config(url: str, schema: str):
    """Alembic config pointed at ``schema`` inside ``url``."""
    from pathlib import Path

    from alembic.config import Config

    root = Path(__file__).resolve().parent.parent
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    # env.py 让显式配置优先于环境变量，正是为了这里能把迁移限定到测试 schema。
    config.set_main_option("sqlalchemy.url", _with_search_path(url, schema))
    return config


def _run_alembic(url: str, schema: str, revision: str) -> None:
    from alembic import command

    command.upgrade(_alembic_config(url, schema), revision)


def alembic_downgrade(url: str, schema: str, revision: str = "base") -> None:
    """Exposed for tests that assert teardown behaviour."""
    from alembic import command

    command.downgrade(_alembic_config(url, schema), revision)


@pytest.fixture
def postgres_connection(postgres_schema):
    """A connection whose ``search_path`` points at the test schema."""
    url = postgres_url()
    assert url is not None
    connection = _connect_postgres(url)
    connection.execute(f"SET search_path TO {postgres_schema}")
    try:
        yield connection
    finally:
        connection.close()
