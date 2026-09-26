from travel_agent.mock_model import MockAgentModel
from travel_agent.requirements import TripRequirements
from travel_agent.workflow import build_requirements_graph


def test_requirements_graph_returns_follow_up_for_first_missing_field():
    graph = build_requirements_graph(MockAgentModel())

    result = graph.invoke({"requirements": TripRequirements(destination="杭州").as_dict()})

    assert result["decision"] == {
        "action": "ASK_USER",
        "message": "请提供出发地。",
        "missing_field": "origin",
    }


def test_requirements_graph_marks_complete_requirements_ready():
    graph = build_requirements_graph(MockAgentModel())

    result = graph.invoke(
        {
            "requirements": TripRequirements(
                origin="北京",
                destination="杭州",
                start_date="2026-10-01",
                end_date="2026-10-04",
                traveler_count=2,
                budget=8000,
            ).as_dict()
        }
    )

    assert result["decision"]["action"] == "READY"
