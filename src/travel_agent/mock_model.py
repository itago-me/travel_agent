from .requirements import AgentDecision, FIELD_LABELS, TripRequirements


class MockAgentModel:
    """Deterministic decision model used before a real model is configured."""

    def decide(self, requirements: TripRequirements) -> AgentDecision:
        missing_fields = requirements.missing_fields()
        if missing_fields:
            field = missing_fields[0]
            return AgentDecision(
                action="ASK_USER",
                missing_field=field,
                message=f"请提供{FIELD_LABELS[field]}。",
            )
        return AgentDecision(
            action="READY",
            message="旅行需求已收集完整，可以开始生成方案。",
        )
