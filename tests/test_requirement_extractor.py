from travel_agent.requirement_extractor import extract_requirements


def test_extract_requirements_reads_common_consultant_message_patterns():
    extracted = extract_requirements(
        "从北京出发，去杭州，2026-10-01到2026-10-04，2人，预算8000元"
    )

    assert extracted == {
        "origin": "北京",
        "destination": "杭州",
        "start_date": "2026-10-01",
        "end_date": "2026-10-04",
        "traveler_count": 2,
        "budget": 8000.0,
    }


def test_extract_requirements_returns_only_fields_found():
    assert extract_requirements("预算调整为6000元") == {"budget": 6000.0}
