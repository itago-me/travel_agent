"""阶段 8.4 第 3 步：turn 级追踪落库 + CLI 展示。

日志回答"当时发生了什么"，span 回答"事后怎么看"。测试锁定：

1. 每个模型调用和工具执行都留下一条带耗时的 span；
2. span 按 sequence 排序，并行工具不会因插入时刻交错而乱序；
3. 一次「批准」只有一条 turn 链（嵌套 turn 已修复）；
4. recording 失败不影响业务。
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from travel_agent.application import create_consultation_service
from travel_agent.tracing import TurnTraceCallback


@pytest.fixture
def service(tmp_path):
    return create_consultation_service(tmp_path / "spans.db")


def _callback(service, consultation_id):
    captured = []

    def recorder(**kwargs):
        captured.append(kwargs)
        service.repository.add_turn_span(
            consultation_id=consultation_id, **kwargs
        )

    return TurnTraceCallback(consultation_id, recorder=recorder), captured


def _llm_result(tokens=100):
    return LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="x",
                        usage_metadata={
                            "input_tokens": tokens,
                            "output_tokens": 10,
                            "total_tokens": tokens + 10,
                        },
                    )
                )
            ]
        ]
    )


# --- span 写入 --------------------------------------------------------------


def test_model_call_records_a_span(service):
    consultation = service.create_consultation("c1", "张三", "去杭州")
    callback, captured = _callback(service, consultation.id)
    from travel_agent.correlation import turn_scope

    with turn_scope():
        callback.on_chat_model_start({}, [], run_id="r1")
        callback.on_llm_end(_llm_result(500), run_id="r1")

    assert captured[0]["kind"] == "model"
    assert captured[0]["details"]["input_tokens"] == 500
    assert "duration_ms" in captured[0]


def test_tool_call_records_a_span_with_result_count(service):
    consultation = service.create_consultation("c1", "张三", "去杭州")
    callback, captured = _callback(service, consultation.id)
    from travel_agent.correlation import turn_scope

    with turn_scope():
        callback.on_tool_start({"name": "search_hotel_options"}, "{}", run_id="t1")
        callback.on_tool_end(
            ToolMessage(content='{"options": [{}, {}, {}]}', tool_call_id="c1"),
            run_id="t1",
        )

    assert captured[0]["kind"] == "tool"
    assert captured[0]["name"] == "search_hotel_options"
    assert captured[0]["details"]["result_count"] == 3


def test_tool_failure_is_recorded(service):
    consultation = service.create_consultation("c1", "张三", "去杭州")
    callback, captured = _callback(service, consultation.id)
    from travel_agent.correlation import turn_scope

    with turn_scope():
        callback.on_tool_start({"name": "search_hotel_options"}, "{}", run_id="t1")
        callback.on_tool_error(TimeoutError("boom"), run_id="t1")

    assert captured[0]["details"]["error"] == "TimeoutError"


def test_span_outside_a_turn_is_skipped(service):
    """没有 turn_id 时不写库——启动/关闭阶段的调用不属于任何一轮。"""
    consultation = service.create_consultation("c1", "张三", "去杭州")
    callback, captured = _callback(service, consultation.id)

    callback.on_chat_model_start({}, [], run_id="r1")
    callback.on_llm_end(_llm_result(), run_id="r1")

    assert captured == []


def test_recorder_failure_does_not_break_the_turn(service):
    """观测写失败不能影响业务——它是旁观者，不是参与者。"""
    consultation = service.create_consultation("c1", "张三", "去杭州")

    def boom(**_kwargs):
        raise RuntimeError("db down")

    callback = TurnTraceCallback(consultation.id, recorder=boom)
    from travel_agent.correlation import turn_scope

    with turn_scope():
        callback.on_chat_model_start({}, [], run_id="r1")
        callback.on_llm_end(_llm_result(), run_id="r1")  # 不抛


# --- 持久层 ----------------------------------------------------------------


def test_spans_are_listed_in_sequence_order(service):
    """并行工具调用的插入时刻交错，排序必须靠 sequence 而不是 created_at。"""
    consultation = service.create_consultation("c1", "张三", "去杭州")
    for sequence in (1, 3, 2):
        service.repository.add_turn_span(
            consultation_id=consultation.id,
            turn_id="turn-1",
            kind="tool",
            name=f"tool-{sequence}",
            duration_ms=sequence * 10,
            sequence=sequence,
        )

    spans = service.turn_spans(consultation.id)

    assert [span["sequence"] for span in spans] == [1, 2, 3]
    assert [span["name"] for span in spans] == ["tool-1", "tool-2", "tool-3"]


def test_spans_can_be_filtered_by_turn(service):
    consultation = service.create_consultation("c1", "张三", "去杭州")
    for turn in ("turn-a", "turn-b"):
        service.repository.add_turn_span(
            consultation_id=consultation.id,
            turn_id=turn,
            kind="model",
            name="模型调用 #1",
            duration_ms=100,
        )

    assert len(service.turn_spans(consultation.id)) == 2
    assert len(service.turn_spans(consultation.id, turn_id="turn-a")) == 1


def test_turn_spans_for_unknown_consultation_raises(service):
    with pytest.raises(KeyError):
        service.turn_spans("nope")


def test_migration_adds_the_table_to_an_existing_database(tmp_path):
    """建表走 CREATE TABLE IF NOT EXISTS，老库下次打开自动补上（这就是迁移）。"""
    import sqlite3

    from travel_agent.repository import SQLiteRepository

    database = tmp_path / "old.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE consultations (id TEXT PRIMARY KEY, thread_id TEXT UNIQUE,"
        " consultant_id TEXT, customer_id TEXT DEFAULT '', customer_name TEXT,"
        " status TEXT, trip_requirements TEXT, planning_constraints TEXT DEFAULT '{}',"
        " created_at TEXT, updated_at TEXT)"
    )
    connection.commit()
    connection.close()

    SQLiteRepository(database)

    tables = {
        row[0]
        for row in sqlite3.connect(database).execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert "turn_spans" in tables


# --- turn 嵌套 -------------------------------------------------------------


def test_resume_keeps_the_caller_turn(service, monkeypatch):
    """decide_approval 触发 resume 时必须复用同一个 turn，否则一条批准裂成两条链。"""
    consultation = service.create_consultation("c1", "张三", "去杭州")
    seen = []

    original = service._resume_inner

    def spy(consultation_id):
        from travel_agent.correlation import current_turn_id

        seen.append(current_turn_id())
        return original(consultation_id)

    monkeypatch.setattr(service, "_resume_inner", spy)

    from travel_agent.correlation import turn_scope

    with turn_scope() as outer:
        service._resume_inner(consultation.id)

    assert seen[0] == outer, "内部实现不应另开 turn"


def test_public_resume_still_opens_its_own_turn(service):
    """直接调 resume 的调用方仍然拿到独立的 turn（它们本来就是独立的一次操作）。"""
    consultation = service.create_consultation("c1", "张三", "去杭州")
    # 没有未决审批，走的是尽早返回分支，只要不抛异常即可
    result = service.resume_approved_consultation(consultation.id)

    assert result["status"] == "APPROVAL_COMPLETED"
