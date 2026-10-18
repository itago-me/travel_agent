"""把当前方案版本注入模型上下文。

真实故障：连续几轮 chat 各产生一个版本后，模型下单时用了**第一轮**的版本 id。
原因是版本列表只存在于数据库和 CLI 输出里，模型上下文里只有对话中出现过的 id，
于是它倾向复用最早见过的那个。``create_mock_booking`` 随即因为"该版本不是已选定
版本"失败。

修复分两层：
1. ``_itinerary_version_state`` 每轮把最新版本列表和选定状态放进 graph state；
2. ``_render_itinerary_versions`` 把它渲染进 system prompt，并显式标出已选定版本。
"""

from travel_agent.agent_context import (
    TravelMemoryContext,
    _render_itinerary_versions,
    build_memory_context,
)


def version(number, version_id, total=None):
    payload = {"budget": {"total": total}} if total is not None else {}
    return {"id": version_id, "version_number": number, "payload": payload}


def context(versions, selected=None):
    return TravelMemoryContext(
        effective_preferences={},
        current_requirements={},
        structured_summary={},
        conversation_summary="",
        itinerary_versions=versions,
        selected_itinerary_version_id=selected,
    )


def test_no_versions_renders_nothing():
    assert _render_itinerary_versions(context([])) == ""


def test_versions_are_listed_with_ids_and_amounts():
    rendered = _render_itinerary_versions(
        context([version(1, "id-1", 2790), version(2, "id-2", 2640)])
    )

    assert "v1 id=id-1" in rendered
    assert "v2 id=id-2" in rendered
    assert "2790" in rendered and "2640" in rendered


def test_selected_version_is_marked():
    rendered = _render_itinerary_versions(
        context([version(1, "id-1", 2790), version(2, "id-2", 2640)], selected="id-2")
    )

    marked_lines = [line for line in rendered.splitlines() if "已选定" in line]
    assert any("id-2" in line for line in marked_lines)
    assert not any("id-1" in line for line in marked_lines)


def test_rendering_warns_when_no_version_is_selected():
    """未选定时要提前告诉模型下单会失败，而不是等工具报错。"""
    rendered = _render_itinerary_versions(context([version(1, "id-1", 2790)]))

    assert "没有任何已选定版本" in rendered
    assert "select-itinerary" in rendered


def test_rendering_states_the_booking_constraint_when_selected():
    rendered = _render_itinerary_versions(
        context([version(1, "id-1", 2790)], selected="id-1")
    )

    assert "id-1" in rendered
    assert "只能使用已选定的版本" in rendered


def test_version_without_budget_is_not_given_a_fake_amount():
    rendered = _render_itinerary_versions(context([version(1, "id-1")]))

    assert "预算未确定" in rendered
    assert "None" not in rendered


def test_memory_context_picks_up_version_state():
    state = {
        "itinerary_versions": [version(1, "id-1", 2790)],
        "selected_itinerary_version_id": "id-1",
    }

    memory = build_memory_context(state)

    assert memory.itinerary_versions[0]["id"] == "id-1"
    assert memory.selected_itinerary_version_id == "id-1"


def test_memory_context_defaults_when_version_state_is_absent():
    """旧 checkpoint 里没有版本字段，读取时不能崩。"""
    memory = build_memory_context({})

    assert memory.itinerary_versions == []
    assert memory.selected_itinerary_version_id is None
