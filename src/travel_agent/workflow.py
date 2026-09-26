from typing import TypedDict
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from .mock_model import MockAgentModel
from .requirements import TripRequirements


class RequirementsState(TypedDict, total=False):
    requirements: dict[str, str | int | float]
    decision: dict[str, str | None]
    model_call_count: int


def build_requirements_graph(model: MockAgentModel, checkpointer=None):
    def evaluate_requirements(
        state: RequirementsState,
    ) -> dict[str, dict[str, str | None] | int]:
        requirements = TripRequirements(**state.get("requirements", {}))
        decision = model.decide(requirements)
        return {
            "decision": {
                "action": decision.action,
                "message": decision.message,
                "missing_field": decision.missing_field,
            },
            "model_call_count": state.get("model_call_count", 0) + 1,
        }

    graph = StateGraph(RequirementsState)
    graph.add_node("evaluate_requirements", evaluate_requirements)
    graph.add_edge(START, "evaluate_requirements")
    graph.add_edge("evaluate_requirements", END)
    return graph.compile(checkpointer=checkpointer)


def run_requirements_graph(
    model: MockAgentModel,
    requirements: dict[str, str | int | float],
    *,
    thread_id: str,
    checkpoint_path: str | Path,
) -> dict[str, object]:
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as checkpointer:
        graph = build_requirements_graph(model, checkpointer=checkpointer)
        return graph.invoke(
            {"requirements": requirements},
            config={"configurable": {"thread_id": thread_id}},
        )
