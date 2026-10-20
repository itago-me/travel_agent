"""阶段 6.2 C：审批金额与版本冻结预算的一致性校验。

真实故障：真实模型走查中，顾问把 2490 改成 2400 后批准，订单按 2400 创建，
而版本冻结金额仍是 2490——两者差额没有任何校验。模型自己连续拒绝了三次
提交才让人注意到这个缺口，说明它此前只存在于模型的自然语言推理里。

设计口径：**记录而非硬拒绝**。顾问改价可能是合理的业务行为（议价、服务费
减免），硬拒绝会阻断正常流程；但审计必须能解释"为什么订单金额与报价不同"。
需要严格模式的部署可以显式开启。
"""

import pytest

from travel_agent.application import create_consultation_service


def make(tmp_path, budget_total=2490, name="t"):
    service = create_consultation_service(tmp_path / f"{name}.db")
    consultation = service.create_consultation("c1", "客户", "去杭州")
    payload = {} if budget_total is None else {"budget": {"total": budget_total}}
    version_id, _ = service.repository.save_itinerary_version(
        consultation.id, payload, [], {}
    )
    service.repository.select_itinerary_version(consultation.id, version_id)
    return service, consultation, version_id


def mismatches(service, consultation_id):
    return [
        event
        for event in service.repository.list_audit_events(consultation_id)
        if event.event_type == "APPROVAL_AMOUNT_MISMATCH"
    ]


# --- 一致的情况不应产生噪音 ------------------------------------------------


def test_matching_amount_records_no_mismatch(tmp_path):
    service, consultation, version_id = make(tmp_path)

    booking = service.repository.create_mock_booking(
        consultation.id, version_id, total_amount=2490, idempotency_key="k1"
    )

    assert booking.total_amount == 2490
    assert mismatches(service, consultation.id) == []


def test_small_difference_within_tolerance_is_not_flagged(tmp_path):
    """5% 以内视为正常议价空间，避免每单都告警。"""
    service, consultation, version_id = make(tmp_path)
    # 2490 的 2% 差额
    service.repository.create_mock_booking(
        consultation.id, version_id, total_amount=2440, idempotency_key="k1"
    )

    assert mismatches(service, consultation.id) == []


# --- 超过容差必须留痕 -------------------------------------------------------


def test_large_difference_is_recorded_but_still_books(tmp_path):
    service, consultation, version_id = make(tmp_path)

    booking = service.repository.create_mock_booking(
        consultation.id, version_id, total_amount=2200, idempotency_key="k1"
    )

    # 默认记录模式：订单照常创建，但审计里有据可查
    assert booking.total_amount == 2200
    events = mismatches(service, consultation.id)
    assert len(events) == 1
    details = events[0].details
    assert details["reason"] == "amount_differs_from_version_budget"
    assert details["booked_total"] == 2200
    assert details["version_total"] == 2490
    assert details["delta"] == -290
    assert details["itinerary_version_id"] == version_id


def test_increase_beyond_tolerance_is_also_recorded(tmp_path):
    """加价同样是差额，不能只盯着降价。"""
    service, consultation, version_id = make(tmp_path)

    service.repository.create_mock_booking(
        consultation.id, version_id, total_amount=3000, idempotency_key="k1"
    )

    details = mismatches(service, consultation.id)[0].details
    assert details["delta"] == 510


def test_zero_version_total_flags_any_nonzero_amount(tmp_path):
    """版本总价为 0 时不能把任何正值都当成"在容差内"。"""
    service, consultation, version_id = make(tmp_path, budget_total=0)

    service.repository.create_mock_booking(
        consultation.id, version_id, total_amount=100, idempotency_key="k1"
    )

    assert len(mismatches(service, consultation.id)) == 1


# --- 无基准的情况 -----------------------------------------------------------


def test_version_without_budget_records_the_missing_baseline(tmp_path):
    """版本没有预算时没有可比基准，记录事实但不判定为异常。"""
    service, consultation, version_id = make(tmp_path, budget_total=None)

    service.repository.create_mock_booking(
        consultation.id, version_id, total_amount=1000, idempotency_key="k1"
    )

    details = mismatches(service, consultation.id)[0].details
    assert details["reason"] == "version_budget_missing"
    assert details["version_total"] is None
    assert details["delta"] is None


def test_missing_budget_is_still_a_single_event(tmp_path):
    service, consultation, version_id = make(tmp_path, budget_total=None)

    service.repository.create_mock_booking(
        consultation.id, version_id, total_amount=1000, idempotency_key="k1"
    )

    assert len(mismatches(service, consultation.id)) == 1


# --- 幂等与严格模式 ---------------------------------------------------------


def test_idempotent_replay_does_not_duplicate_the_mismatch_event(tmp_path):
    service, consultation, version_id = make(tmp_path)
    for _ in range(2):
        service.repository.create_mock_booking(
            consultation.id, version_id, total_amount=2200, idempotency_key="k1"
        )

    assert len(mismatches(service, consultation.id)) == 1


def test_已存在的订单不会被事后收紧的容差判为非法(tmp_path, monkeypatch):
    """先按宽松策略建单，再把容差设为 0 重放，不能因此报错。

    这正是把校验放在幂等检查之后的原因：顾问调整配置不应让历史订单失效。
    """
    service, consultation, version_id = make(tmp_path)
    first = service.repository.create_mock_booking(
        consultation.id, version_id, total_amount=2200, idempotency_key="k1"
    )
    monkeypatch.setenv("TRAVEL_AGENT_BOOKING_AMOUNT_TOLERANCE", "0")

    replay = service.repository.create_mock_booking(
        consultation.id, version_id, total_amount=2200, idempotency_key="k1"
    )

    assert replay.id == first.id


def test_strict_mode_blocks_an_over_tolerance_booking(tmp_path, monkeypatch):
    service, consultation, version_id = make(tmp_path)
    monkeypatch.setenv("TRAVEL_AGENT_BOOKING_AMOUNT_STRICT", "1")

    with pytest.raises(ValueError, match="differs from the itinerary version budget"):
        service.repository.create_mock_booking(
            consultation.id, version_id, total_amount=2200, idempotency_key="k1"
        )

    assert service.repository.list_mock_bookings(consultation.id) == []


def test_strict_mode_still_allows_a_matching_amount(tmp_path, monkeypatch):
    service, consultation, version_id = make(tmp_path)
    monkeypatch.setenv("TRAVEL_AGENT_BOOKING_AMOUNT_STRICT", "1")

    booking = service.repository.create_mock_booking(
        consultation.id, version_id, total_amount=2490, idempotency_key="k1"
    )

    assert booking.total_amount == 2490


def test_tolerance_is_configurable(tmp_path, monkeypatch):
    """容差设为 0 时，1 元差额也算不一致。"""
    service, consultation, version_id = make(tmp_path)
    monkeypatch.setenv("TRAVEL_AGENT_BOOKING_AMOUNT_TOLERANCE", "0")

    service.repository.create_mock_booking(
        consultation.id, version_id, total_amount=2489, idempotency_key="k1"
    )

    assert len(mismatches(service, consultation.id)) == 1


def test_mismatch_event_and_booking_share_one_transaction(tmp_path, monkeypatch):
    """严格模式拒绝时不能留下任何痕迹：订单和差额事件都不该存在。"""
    service, consultation, version_id = make(tmp_path)
    monkeypatch.setenv("TRAVEL_AGENT_BOOKING_AMOUNT_STRICT", "1")

    with pytest.raises(ValueError):
        service.repository.create_mock_booking(
            consultation.id, version_id, total_amount=100, idempotency_key="k1"
        )

    assert service.repository.list_mock_bookings(consultation.id) == []
    assert mismatches(service, consultation.id) == []
