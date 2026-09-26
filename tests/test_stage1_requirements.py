from travel_agent.mock_model import MockAgentModel
from travel_agent.requirements import TripRequirements


def test_trip_requirements_reports_missing_fields_in_business_order():
    requirements = TripRequirements(destination="杭州")

    assert requirements.missing_fields() == [
        "origin",
        "start_date",
        "end_date",
        "traveler_count",
        "budget",
    ]


def test_mock_model_asks_for_the_first_missing_requirement():
    model = MockAgentModel()

    decision = model.decide(TripRequirements(destination="杭州"))

    assert decision.action == "ASK_USER"
    assert decision.missing_field == "origin"
    assert decision.message == "请提供出发地。"


def test_mock_model_is_ready_when_all_required_fields_are_present():
    model = MockAgentModel()
    requirements = TripRequirements(
        origin="北京",
        destination="杭州",
        start_date="2026-10-01",
        end_date="2026-10-04",
        traveler_count=2,
        budget=8000,
    )

    decision = model.decide(requirements)

    assert decision.action == "READY"
    assert decision.missing_field is None
    assert decision.message == "旅行需求已收集完整，可以开始生成方案。"
