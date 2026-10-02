from travel_agent.planning_revision import parse_planning_revision


def test_parse_revision_extracts_budget_transport_and_hotel_constraints():
    assert parse_planning_revision("预算调整为6000元，优先高铁，酒店换成评分更高的") == {
        "budget": 6000.0,
        "transport_mode": "train",
        "hotel_preference": "highest_rating",
    }
