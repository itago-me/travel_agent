import pytest

from travel_agent.domain import ApprovalDecision, ApprovalStatus, BookingStatus
from travel_agent.repository import SQLiteRepository
from travel_agent.service import ConsultationService


def prepared_repository(tmp_path):
    repository = SQLiteRepository(tmp_path / "travel.db")
    service = ConsultationService(repository)
    consultation = service.create_consultation(
        "consultant-1", "张先生", "去杭州", customer_id="customer-1"
    )
    version_id, _ = repository.save_itinerary_version(
        consultation.id,
        {"budget": {"total": 7800}},
        [],
        {},
    )
    repository.select_itinerary_version(consultation.id, version_id)
    return repository, consultation, version_id


def test_approval_request_persists_sensitive_action_and_decision(tmp_path):
    repository, consultation, version_id = prepared_repository(tmp_path)
    approval = repository.create_approval_request(
        consultation.id,
        consultation.thread_id,
        "create_mock_booking",
        {
            "consultation_id": consultation.id,
            "itinerary_version_id": version_id,
            "total_amount": 7800,
            "idempotency_key": "booking-1",
        },
        requested_by="agent",
    )

    assert approval.status is ApprovalStatus.PENDING
    assert approval.risk_level == "SENSITIVE"
    assert repository.get_approval_request(approval.id) == approval

    decided = repository.decide_approval_request(
        approval.id,
        ApprovalDecision.APPROVE,
        decided_by="consultant-1",
        reason="客户已确认",
    )
    assert decided.status is ApprovalStatus.APPROVED
    assert decided.decision is ApprovalDecision.APPROVE
    assert decided.decided_at is not None

    with pytest.raises(ValueError, match="already decided"):
        repository.decide_approval_request(
            approval.id,
            ApprovalDecision.REJECT,
            decided_by="consultant-1",
            reason="重复决定",
        )


def test_rejection_requires_reason_and_is_customer_consultation_scoped(tmp_path):
    repository, consultation, _ = prepared_repository(tmp_path)
    approval = repository.create_approval_request(
        consultation.id,
        consultation.thread_id,
        "cancel_mock_booking",
        {"booking_id": "booking-1"},
        requested_by="agent",
    )

    with pytest.raises(ValueError, match="reason"):
        repository.decide_approval_request(
            approval.id,
            ApprovalDecision.REJECT,
            decided_by="consultant-1",
        )

    rejected = repository.decide_approval_request(
        approval.id,
        ApprovalDecision.REJECT,
        decided_by="consultant-1",
        reason="客户暂不预订",
    )
    assert rejected.status is ApprovalStatus.REJECTED
    assert repository.list_approval_requests(consultation.id) == [rejected]


def test_edit_decision_requires_and_persists_replacement_arguments(tmp_path):
    repository, consultation, version_id = prepared_repository(tmp_path)
    approval = repository.create_approval_request(
        consultation.id,
        consultation.thread_id,
        "create_mock_booking",
        {"total_amount": 7800, "itinerary_version_id": version_id},
        requested_by="agent",
    )
    with pytest.raises(ValueError, match="edited_arguments"):
        repository.decide_approval_request(
            approval.id, ApprovalDecision.EDIT, decided_by="consultant-1"
        )

    edited = repository.decide_approval_request(
        approval.id,
        ApprovalDecision.EDIT,
        decided_by="consultant-1",
        reason="金额按最终报价调整",
        edited_arguments={"total_amount": 7600, "itinerary_version_id": version_id},
    )
    assert edited.status is ApprovalStatus.APPROVED
    assert edited.decision is ApprovalDecision.EDIT
    assert edited.edited_arguments["total_amount"] == 7600


def test_mock_booking_is_idempotent_and_cancel_is_idempotent(tmp_path):
    repository, consultation, version_id = prepared_repository(tmp_path)
    first = repository.create_mock_booking(
        consultation.id,
        version_id,
        total_amount=7800,
        idempotency_key="create-booking-1",
    )
    second = repository.create_mock_booking(
        consultation.id,
        version_id,
        total_amount=7800,
        idempotency_key="create-booking-1",
    )
    assert first == second
    assert first.status is BookingStatus.CREATED
    assert len(repository.list_mock_bookings(consultation.id)) == 1

    with pytest.raises(ValueError, match="different booking payload"):
        repository.create_mock_booking(
            consultation.id,
            version_id,
            total_amount=7900,
            idempotency_key="create-booking-1",
        )

    cancelled = repository.cancel_mock_booking(
        consultation.id,
        first.id,
        reason="客户取消",
        idempotency_key="cancel-booking-1",
    )
    repeated = repository.cancel_mock_booking(
        consultation.id,
        first.id,
        reason="客户取消",
        idempotency_key="cancel-booking-1",
    )
    assert cancelled == repeated
    assert cancelled.status is BookingStatus.CANCELLED

