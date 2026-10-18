"""预订前必须选定方案版本：错误信息要能直接告诉顾问怎么修。

真实故障：报错只有一句 `itinerary version must be selected before booking`，
既没说请求的是哪个版本、当前选的是哪个，也没说该跑什么命令。顾问只能自己
去翻库。这里把三类失败都做成可操作的信息。
"""

import pytest

from travel_agent.application import create_consultation_service


def make(tmp_path, versions=2):
    service = create_consultation_service(tmp_path / "travel.db")
    consultation = service.create_consultation("c1", "客户", "去杭州")
    ids = []
    for index in range(versions):
        version_id, _ = service.repository.save_itinerary_version(
            consultation.id, {"budget": {"total": 2000 + index * 100}}, [], {}
        )
        ids.append(version_id)
    return service, consultation, ids


def test_unselected_version_error_names_both_sides_and_the_fix(tmp_path):
    service, consultation, ids = make(tmp_path)

    with pytest.raises(ValueError) as error:
        service.repository.create_mock_booking(
            consultation.id, ids[0], total_amount=2000, idempotency_key="k1"
        )

    message = str(error.value)
    # 说清请求的版本、当前没选、以及有哪些可选
    assert ids[0] in message
    assert "no version is selected yet" in message
    assert f"v1={ids[0]}" in message and f"v2={ids[1]}" in message
    # 直接给出修复命令
    assert f"select-itinerary {consultation.id} {ids[0]}" in message


def test_mismatched_selection_error_reports_the_selected_version(tmp_path):
    service, consultation, ids = make(tmp_path)
    service.repository.select_itinerary_version(consultation.id, ids[1])

    with pytest.raises(ValueError) as error:
        service.repository.create_mock_booking(
            consultation.id, ids[0], total_amount=2000, idempotency_key="k1"
        )

    message = str(error.value)
    assert f"requested {ids[0]}" in message
    assert f"selected version is {ids[1]}" in message
    assert f"select-itinerary {consultation.id} {ids[0]}" in message


def test_unknown_version_error_lists_available_versions(tmp_path):
    service, consultation, ids = make(tmp_path)

    with pytest.raises(KeyError) as error:
        service.repository.create_mock_booking(
            consultation.id, "missing-version", total_amount=2000, idempotency_key="k1"
        )

    message = str(error.value)
    assert "missing-version" in message
    assert f"v1={ids[0]}" in message


def test_unknown_version_with_no_versions_suggests_running_a_chat_turn(tmp_path):
    service = create_consultation_service(tmp_path / "travel.db")
    consultation = service.create_consultation("c1", "客户", "去杭州")

    with pytest.raises(KeyError) as error:
        service.repository.create_mock_booking(
            consultation.id, "missing-version", total_amount=2000, idempotency_key="k1"
        )

    assert "no itinerary version yet" in str(error.value)
    assert "chat turn" in str(error.value)


def test_selected_version_books_successfully(tmp_path):
    service, consultation, ids = make(tmp_path)
    service.repository.select_itinerary_version(consultation.id, ids[0])

    booking = service.repository.create_mock_booking(
        consultation.id, ids[0], total_amount=2000, idempotency_key="k1"
    )

    assert booking.itinerary_version_id == ids[0]
    assert booking.total_amount == 2000
