from travel_agent.mock_model import MockAgentModel
from travel_agent.requirements import TripRequirements
from travel_agent.workflow import run_requirements_graph


def test_requirements_graph_restores_state_for_the_same_thread(tmp_path):
    checkpoint_path = tmp_path / "checkpoints.sqlite"
    model = MockAgentModel()

    first = run_requirements_graph(
        model,
        TripRequirements(destination="杭州").as_dict(),
        thread_id="consultation-thread-1",
        checkpoint_path=checkpoint_path,
    )
    second = run_requirements_graph(
        model,
        TripRequirements(
            origin="北京",
            destination="杭州",
            start_date="2026-10-01",
            end_date="2026-10-04",
            traveler_count=2,
            budget=8000,
        ).as_dict(),
        thread_id="consultation-thread-1",
        checkpoint_path=checkpoint_path,
    )

    assert first["decision"]["action"] == "ASK_USER"
    assert first["model_call_count"] == 1
    assert second["decision"]["action"] == "READY"
    assert second["model_call_count"] == 2


def test_requirements_graph_keeps_checkpoint_threads_isolated(tmp_path):
    checkpoint_path = tmp_path / "checkpoints.sqlite"
    result = run_requirements_graph(
        MockAgentModel(),
        TripRequirements(destination="杭州").as_dict(),
        thread_id="another-thread",
        checkpoint_path=checkpoint_path,
    )

    assert result["model_call_count"] == 1
