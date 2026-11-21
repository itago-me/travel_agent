"""阶段 9 C2：Postgres 路径端到端与并发验证。

C1 证明了「零件都能装」——仓储、工具、checkpointer 都能指向 Postgres。
C2 要证明**整机通电**：

- **A 组（正确性）**：完整一轮 ``chat_consultation`` 在 Postgres 仓储下跑完，
  消息、审计、方案版本全部落库，两轮之间 usage 正确累加。
- **B 组（checkpoint 持久化）**：用真正的 ``AsyncPostgresSaver`` 跑一个小图，
  再用**独立的连接**读回来——这才是「重启后对话能继续」的真实证明。
- **D 组（多写入者）**：两个独立进程同时写同一个库，不丢数据、不报错。
  这是迁移的真正收益所在：SQLite 靠文件锁串行化写，多实例部署会互相排队。

C 组（并发延迟对比）放在 ``experiments/db_concurrency_demo.py``，那里更适合
跑长时间基准；这里只放确定性的断言。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from travel_agent.postgres_repository import PostgresRepository, plain_dsn
from conftest import postgres_url


# ---------------------------------------------------------------------------
# 公共 fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def pg_schema():
    """建一个隔离 schema 并在其上跑 Alembic；没有 Postgres 时 skip。"""
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
        yield url, schema
    finally:
        drop_schema(admin, schema)
        admin.close()


@pytest.fixture
def pg_repo(pg_schema):
    url, schema = pg_schema
    repository = PostgresRepository(plain_dsn(url), schema=schema)
    try:
        yield repository
    finally:
        repository.close()


class _RecordingAgent:
    """答一轮就返回的假 Agent，记录调用次数以便断言 thread 复用。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def ainvoke(self, payload, *, config):
        self.calls.append(config)
        turn = len(self.calls)
        return {
            "messages": [
                HumanMessage(content="用户这一轮的话"),
                AIMessage(
                    content=f"第 {turn} 轮回复",
                    usage_metadata={
                        "input_tokens": 100 * turn,
                        "output_tokens": 20 * turn,
                        "total_tokens": 120 * turn,
                    },
                ),
            ]
        }

    async def aget_state(self, config):
        return SimpleNamespace(values={"messages": []}, next=(), interrupts=())


@pytest.fixture
def service(pg_repo, tmp_path, monkeypatch):
    """Postgres 业务仓储 + SQLite checkpoint 的服务（C1 的混合形态）。"""
    from travel_agent.service import ConsultationService

    monkeypatch.setattr(
        "travel_agent.service.load_agent_settings",
        lambda *_a, **_k: SimpleNamespace(
            model_mode="openai_compatible",
            model_name="demo",
            api_key=None,
            base_url=None,
        ),
    )
    return ConsultationService(pg_repo, agent_database=tmp_path / "tools.db")


# ---------------------------------------------------------------------------
# A 组：完整一轮对话落在 Postgres
# ---------------------------------------------------------------------------


def test_one_full_turn_persists_everything_to_postgres(service, pg_repo, monkeypatch):
    agent = _RecordingAgent()
    monkeypatch.setattr(
        "travel_agent.service.create_agent_app", lambda *a, **k: agent
    )
    consultation = service.create_consultation("c1", "测试客户", "我想去杭州")

    result = service.chat_consultation(consultation.id, "查一下酒店")

    assert result["status"] == "COMPLETED"
    assert result["answer"] == "第 1 轮回复"

    # 全部从 Postgres 读回——不是内存里的返回值
    stored = pg_repo.get_consultation(consultation.id)
    roles = [message.role for message in stored.messages]
    assert roles == ["user", "user", "assistant"]

    events = pg_repo.list_audit_events(consultation.id)
    completed = [e for e in events if e.event_type == "AGENT_CHAT_COMPLETED"]
    assert len(completed) == 1
    assert completed[0].details["usage"]["input_tokens"] == 100
    assert completed[0].details["usage"]["reported"] is True


def test_two_turns_accumulate_usage_in_postgres(service, pg_repo, monkeypatch):
    agent = _RecordingAgent()
    monkeypatch.setattr(
        "travel_agent.service.create_agent_app", lambda *a, **k: agent
    )
    consultation = service.create_consultation("c1", "测试客户", "我想去杭州")

    service.chat_consultation(consultation.id, "第一轮")
    service.chat_consultation(consultation.id, "第二轮")

    totals = pg_repo.consult_usage_totals(consultation.id)
    assert totals["turns"] == 2
    assert totals["input_tokens"] == 100 + 200
    assert totals["output_tokens"] == 20 + 40


def test_turn_spans_are_written_to_postgres(pg_repo, tmp_path):
    """追踪表也要落 Postgres，否则 /metrics 与 trace 会读空。"""
    from travel_agent.service import ConsultationService

    service = ConsultationService(pg_repo, agent_database=tmp_path / "tools.db")
    consultation = service.create_consultation("c1", "测试客户", "想去杭州")

    record = service._span_recorder(consultation.id)
    record(
        turn_id="turn-1",
        kind="model",
        name="模型调用 #1",
        duration_ms=42,
        details={},
    )

    spans = service.turn_spans(consultation.id, turn_id="turn-1")
    assert len(spans) == 1
    assert spans[0]["name"] == "模型调用 #1"
    assert spans[0]["duration_ms"] == 42


# ---------------------------------------------------------------------------
# B 组：checkpoint 真的持久化在 Postgres
# ---------------------------------------------------------------------------


async def _open_saver(dsn: str, schema: str):
    """把一个 saver 绑到指定 schema。

    ``search_path`` 用 psycopg 的 ``options=`` 参数传，**不要拼进 URL**：
    psycopg 的 URI 解析器会拒绝取值里未编码的 ``=``
    （``extra key/value separator "=" in URI query parameter``），而编码成
    ``%3D`` 又会被 Alembic 那边的 configparser 拒绝（``%`` 是插值语法）。
    两条路径各有各的禁忌，用连接参数是最不容易踩坑的写法。
    """
    import psycopg
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    connection = await psycopg.AsyncConnection.connect(
        dsn, autocommit=True, options=f"-csearch_path={schema}"
    )
    return AsyncPostgresSaver(connection), connection


def test_checkpoint_survives_a_fresh_connection(pg_schema):
    """写完 checkpoint 后用**新连接**读回——等价于「进程重启对话还在」。"""
    from langgraph.graph import END, START, StateGraph

    url, schema = pg_schema
    dsn = plain_dsn(url)

    async def run_once():
        saver, connection = await _open_saver(dsn, schema)
        try:
            await saver.setup()
            graph = StateGraph(dict)
            graph.add_node("echo", lambda state: {"seen": state.get("seen", 0) + 1})
            graph.add_edge(START, "echo")
            graph.add_edge("echo", END)
            app = graph.compile(checkpointer=saver)
            await app.ainvoke({"seen": 0}, {"configurable": {"thread_id": "t-1"}})
        finally:
            await connection.close()

    asyncio.run(run_once())

    async def read_back():
        saver, connection = await _open_saver(dsn, schema)
        try:
            graph = StateGraph(dict)
            graph.add_node("echo", lambda state: state)
            graph.add_edge(START, "echo")
            graph.add_edge("echo", END)
            app = graph.compile(checkpointer=saver)
            return await app.aget_state({"configurable": {"thread_id": "t-1"}})
        finally:
            await connection.close()

    state = asyncio.run(read_back())

    assert state.values.get("seen") == 1


# ---------------------------------------------------------------------------
# D 组：多个独立写入者
# ---------------------------------------------------------------------------


def test_two_independent_writers_do_not_lose_data(pg_schema):
    """两个各自持有连接池的仓储同时写，行数必须对得上。

    SQLite 下这会靠文件锁串行化——不丢数据但会排队；Postgres 下各自写各自的，
    这是多实例部署的前提。
    """
    from concurrent.futures import ThreadPoolExecutor

    from travel_agent.domain import AuditEvent, Consultation

    url, schema = pg_schema
    dsn = plain_dsn(url)

    first = PostgresRepository(dsn, schema=schema)
    second = PostgresRepository(dsn, schema=schema)
    try:
        consultation = Consultation(
            id=str(uuid4()),
            thread_id=str(uuid4()),
            consultant_id="c1",
            customer_name="并发客户",
            status="OPEN",
            trip_requirements={},
        )
        first.create_consultation(consultation)

        def write(index: int) -> None:
            repo = first if index % 2 == 0 else second
            repo.add_audit_event(
                AuditEvent(consultation.id, f"EVENT_{index}", {"i": index})
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write, range(40)))

        events = first.list_audit_events(consultation.id)
        assert len(events) == 40
    finally:
        first.close()
        second.close()
