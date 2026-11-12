"""Shared test setup: keep the suite independent of ambient configuration.

``load_agent_settings`` calls ``dotenv.load_dotenv``, which writes the values
into ``os.environ`` for the rest of the process. A developer's ``.env`` can
therefore change how unrelated tests behave, depending only on execution order:
a ``TRAVEL_AGENT_API_TOKEN`` in ``.env`` made every later API test return 401
(observed - the failures pointed at auth instead of at the test that triggered
loading).

Clearing the HTTP token here makes every test start from the same state.
Tests that need auth set it explicitly through ``monkeypatch``, which runs after
this fixture.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_from_dotenv(monkeypatch):
    """Start each test without an HTTP auth token, whatever ``.env`` says."""
    monkeypatch.delenv("TRAVEL_AGENT_API_TOKEN", raising=False)
